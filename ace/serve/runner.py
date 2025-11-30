"""Online server for test-time sequential adaptation.

Implements the ACE paper's online mode:
"In the online setting, no ground-truth labels are available. The reflector
relies on execution feedback (test outputs, logs, errors) to derive insights."
"""

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI

from ace.core.config import ACEConfig, load_config
from ace.core.merge import Delta as MergeDelta
from ace.core.merge import apply_delta
from ace.core.retrieve import Retriever
from ace.core.schema import Playbook
from ace.core.storage.store_adapter import Store
from ace.curator.curator import curate
from ace.refine.runner import refine as run_refine
from ace.reflector.reflector import Reflector
from ace.reflector.schema import Reflection

from .schema import (
    AdaptationMode,
    FeedbackRequest,
    FeedbackResponse,
    OnlineStats,
    RetrieveRequest,
    RetrieveResponse,
    WarmupSource,
)

logger = logging.getLogger(__name__)


class OnlineServer:
    """HTTP server for online (test-time) adaptation.

    Key differences from offline mode:
    - No ground-truth labels; uses execution feedback only
    - Sequential processing (one request at a time)
    - Immediate adaptation after each feedback
    - No epochs; continuous learning

    Supports warm-start by preloading a playbook before accepting queries.
    Paper Table 3 shows 'ReAct + ACE + offline warmup' beats cold-start.
    """

    def __init__(
        self,
        store: Store | None = None,
        reflector: Reflector | None = None,
        retriever: Retriever | None = None,
        auto_adapt: bool = True,
        warmup_path: str | Path | None = None,
        auto_refine_every: int = 0,
        max_bullets: int | None = None,
        config: ACEConfig | None = None,
    ):
        """Initialize the online server.

        Args:
            store: Playbook store (loads from config if None)
            reflector: Reflector instance (creates default if None)
            retriever: Retriever instance (creates default if None)
            auto_adapt: Whether to auto-adapt on each feedback (default True)
            warmup_path: Path to a playbook JSON file for warm-start
            auto_refine_every: Run refine every N deltas (0 = disabled)
            max_bullets: Max bullets before triggering refine (overrides config)
            config: ACEConfig instance (loads if None)
        """
        if config is None:
            config = load_config()
        self._config = config

        if store is None:
            store = Store(config.database.url)
        self.store = store
        self.reflector = reflector or Reflector()
        self.retriever = retriever or Retriever(store)
        self.auto_adapt = auto_adapt
        self.mode = AdaptationMode.ONLINE

        self.auto_refine_every = auto_refine_every
        self.max_bullets = max_bullets if max_bullets is not None else config.retrieval.max_bullets
        self._delta_count_since_refine = 0

        self.session_id = str(uuid.uuid4())[:8]

        warmup_source = WarmupSource.NONE
        warmup_bullets_loaded = 0
        warmup_playbook_version = 0

        if warmup_path:
            warmup_source, warmup_bullets_loaded, warmup_playbook_version = (
                self._load_warmup_playbook(warmup_path)
            )
        else:
            existing = self.store.load_playbook()
            if existing.bullets:
                warmup_source = WarmupSource.DATABASE
                warmup_bullets_loaded = len(existing.bullets)
                warmup_playbook_version = existing.version

        self.stats = OnlineStats(
            session_id=self.session_id,
            warmup_source=warmup_source,
            warmup_bullets_loaded=warmup_bullets_loaded,
            warmup_playbook_version=warmup_playbook_version,
        )

    def _load_warmup_playbook(
        self, warmup_path: str | Path
    ) -> tuple[WarmupSource, int, int]:
        """Load a playbook from file for warm-start.

        Args:
            warmup_path: Path to the playbook JSON file

        Returns:
            Tuple of (warmup_source, bullets_loaded, playbook_version)

        Raises:
            FileNotFoundError: If warmup file doesn't exist
            ValueError: If warmup file is invalid JSON or schema
        """
        path = Path(warmup_path)
        if not path.exists():
            raise FileNotFoundError(f"Warmup playbook not found: {path}")

        with open(path) as f:
            data = json.load(f)

        playbook = Playbook.model_validate(data)
        self.store.load_playbook_data(playbook)

        logger.info(
            f"Warm-start: loaded {len(playbook.bullets)} bullets "
            f"(version {playbook.version}) from {path}"
        )

        return WarmupSource.FILE, len(playbook.bullets), playbook.version

    def retrieve(self, query: str, top_k: int = 24) -> RetrieveResponse:
        """Retrieve bullets for a query.

        Args:
            query: The query to retrieve bullets for
            top_k: Number of bullets to retrieve

        Returns:
            RetrieveResponse with bullets and timing
        """
        start = time.time()
        bullets = self.retriever.retrieve(query, top_k=top_k)
        elapsed_ms = (time.time() - start) * 1000

        return RetrieveResponse(
            bullets=[b.model_dump() for b in bullets],
            retrieval_ms=round(elapsed_ms, 2),
        )

    def process_feedback(self, request: FeedbackRequest) -> FeedbackResponse:
        """Process execution feedback and adapt the playbook.

        In online mode, we rely on execution feedback (test_output, logs,
        execution_success) rather than ground-truth labels.

        Args:
            request: Feedback request with execution results

        Returns:
            FeedbackResponse with adaptation results
        """
        start = time.time()

        try:
            reflection = self.reflector.reflect(
                query=request.query,
                retrieved_bullet_ids=request.retrieved_bullet_ids,
                code_diff=request.code_diff,
                test_output=request.test_output,
                logs=request.logs,
                env_meta=request.env_meta,
            )

            playbook = self.store.load_playbook()
            delta = curate(reflection, existing_bullets=playbook.bullets)

            if not delta.ops:
                elapsed_ms = (time.time() - start) * 1000
                self.stats.requests_processed += 1
                return FeedbackResponse(
                    success=True,
                    ops_applied=0,
                    playbook_version=playbook.version,
                    adaptation_ms=round(elapsed_ms, 2),
                    message="No adaptation needed",
                )

            if self.auto_adapt:
                merge_delta = MergeDelta.from_dict(delta.model_dump())
                new_playbook = apply_delta(playbook, merge_delta, self.store)
                version = new_playbook.version
                self._delta_count_since_refine += 1
            else:
                version = playbook.version

            elapsed_ms = (time.time() - start) * 1000
            ops_count = len(delta.ops)

            self.stats.requests_processed += 1
            self.stats.total_ops_applied += ops_count
            self._update_avg_adaptation_ms(elapsed_ms)

            for op in delta.ops:
                if hasattr(op, "op") and op.op == "INCR_HELPFUL":
                    self.stats.helpful_feedback_count += 1
                elif hasattr(op, "op") and op.op == "INCR_HARMFUL":
                    self.stats.harmful_feedback_count += 1

            logger.info(
                f"Online adaptation: {ops_count} ops, "
                f"version {playbook.version} -> {version}, "
                f"{elapsed_ms:.1f}ms"
            )

            if self.auto_adapt:
                self._maybe_auto_refine()

            return FeedbackResponse(
                success=True,
                ops_applied=ops_count,
                playbook_version=version,
                adaptation_ms=round(elapsed_ms, 2),
                message=f"Applied {ops_count} operations",
            )

        except Exception as e:
            elapsed_ms = (time.time() - start) * 1000
            logger.error(f"Online adaptation error: {e}")
            return FeedbackResponse(
                success=False,
                adaptation_ms=round(elapsed_ms, 2),
                message=str(e),
            )

    def _update_avg_adaptation_ms(self, new_ms: float) -> None:
        """Update running average of adaptation time."""
        n = self.stats.requests_processed
        if n == 1:
            self.stats.avg_adaptation_ms = new_ms
        else:
            self.stats.avg_adaptation_ms = (
                self.stats.avg_adaptation_ms * (n - 1) + new_ms
            ) / n

    def _maybe_auto_refine(self) -> None:
        """Check if auto-refine should trigger and run if so.

        Triggers when:
        - auto_refine_every > 0 and delta count reached threshold
        - max_bullets > 0 and bullet count exceeds max_bullets
        """
        playbook = self.store.load_playbook()
        bullet_count = len(playbook.bullets)

        should_refine = False

        if self.auto_refine_every > 0 and self._delta_count_since_refine >= self.auto_refine_every:
            should_refine = True
            logger.info(
                f"Auto-refine triggered: {self._delta_count_since_refine} deltas "
                f"(threshold: {self.auto_refine_every})"
            )

        if self.max_bullets > 0 and bullet_count > self.max_bullets:
            should_refine = True
            logger.info(
                f"Auto-refine triggered: {bullet_count} bullets > max {self.max_bullets}"
            )

        if not should_refine:
            return

        original_ids = {b.id for b in playbook.bullets}

        empty_reflection = Reflection()
        result = run_refine(
            empty_reflection,
            playbook,
            threshold=self._config.refine.threshold,
        )

        refined_ids = {b.id for b in playbook.bullets}
        removed_ids = original_ids - refined_ids

        for bullet_id in removed_ids:
            self.store.delete_bullet(bullet_id)

        for bullet in playbook.bullets:
            self.store.save_bullet(bullet)

        self._delta_count_since_refine = 0

        self.stats.auto_refine_runs += 1
        self.stats.auto_refine_merged += result.merged
        self.stats.auto_refine_archived += result.archived

        logger.info(
            f"Auto-refine complete: merged={result.merged}, archived={result.archived}, "
            f"removed={len(removed_ids)}, bullets={len(playbook.bullets)}"
        )

    def get_stats(self) -> OnlineStats:
        """Get current session statistics."""
        return self.stats

    def get_playbook_version(self) -> int:
        """Get current playbook version."""
        return self.store.load_playbook().version


def create_app(
    auto_adapt: bool = True,
    store: Store | None = None,
    warmup_path: str | Path | None = None,
    auto_refine_every: int = 0,
    max_bullets: int | None = None,
) -> FastAPI:
    """Create FastAPI application for online serving.

    Args:
        auto_adapt: Whether to automatically adapt on feedback
        store: Optional store instance
        warmup_path: Path to playbook JSON file for warm-start
        auto_refine_every: Run refine every N deltas (0 = disabled)
        max_bullets: Max bullets before triggering refine (overrides config)

    Returns:
        FastAPI app instance
    """
    server_instance: list[OnlineServer] = []

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        instance = OnlineServer(
            store=store,
            auto_adapt=auto_adapt,
            warmup_path=warmup_path,
            auto_refine_every=auto_refine_every,
            max_bullets=max_bullets,
        )
        server_instance.append(instance)
        warmup_info = (
            f", warmup={instance.stats.warmup_source.value}"
            f" ({instance.stats.warmup_bullets_loaded} bullets)"
            if instance.stats.warmup_source != WarmupSource.NONE
            else ""
        )
        logger.info(
            f"Online server started: session={instance.session_id}, "
            f"auto_adapt={auto_adapt}{warmup_info}"
        )
        yield
        server_instance.clear()
        logger.info("Online server shutting down")

    def get_server() -> OnlineServer:
        if not server_instance:
            raise RuntimeError("Server not initialized")
        return server_instance[0]

    app = FastAPI(
        title="ACE Online Server",
        description="Test-time sequential adaptation using execution feedback",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "mode": "online"}

    @app.post("/retrieve")
    async def retrieve(request: RetrieveRequest) -> RetrieveResponse:
        """Retrieve bullets for a query."""
        return get_server().retrieve(request.query, request.top_k)

    @app.post("/feedback")
    async def feedback(request: FeedbackRequest) -> FeedbackResponse:
        """Process execution feedback and adapt playbook."""
        return get_server().process_feedback(request)

    @app.get("/stats")
    async def stats() -> dict[str, Any]:
        """Get session statistics."""
        return get_server().get_stats().model_dump()

    @app.get("/playbook/version")
    async def playbook_version() -> dict[str, int]:
        """Get current playbook version."""
        return {"version": get_server().get_playbook_version()}

    return app


def run_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    auto_adapt: bool = True,
    reload: bool = False,
    warmup_path: str | Path | None = None,
    auto_refine_every: int = 0,
    max_bullets: int | None = None,
) -> None:
    """Run the online server.

    Args:
        host: Host to bind to
        port: Port to bind to
        auto_adapt: Whether to auto-adapt on feedback
        reload: Whether to enable hot reload
        warmup_path: Path to playbook JSON file for warm-start
        auto_refine_every: Run refine every N deltas (0 = disabled)
        max_bullets: Max bullets before triggering refine (overrides config)
    """
    import uvicorn

    warmup_msg = f" (warmup: {warmup_path})" if warmup_path else ""
    refine_msg = ""
    if auto_refine_every > 0:
        refine_msg += f", auto-refine every {auto_refine_every} deltas"
    if max_bullets is not None:
        refine_msg += f", max bullets {max_bullets}"
    logger.info(f"Starting ACE online server on {host}:{port}{warmup_msg}{refine_msg}")
    app = create_app(
        auto_adapt=auto_adapt,
        warmup_path=warmup_path,
        auto_refine_every=auto_refine_every,
        max_bullets=max_bullets,
    )
    uvicorn.run(app, host=host, port=port, reload=reload)
