"""Online server for test-time sequential adaptation.

Implements the ACE paper's online mode:
"In the online setting, no ground-truth labels are available. The reflector
relies on execution feedback (test outputs, logs, errors) to derive insights."
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from ace.core.config import load_config
from ace.core.merge import Delta as MergeDelta
from ace.core.merge import apply_delta
from ace.core.retrieve import Retriever
from ace.core.storage.store_adapter import Store
from ace.curator.curator import curate
from ace.reflector.reflector import Reflector

from .schema import (
    AdaptationMode,
    FeedbackRequest,
    FeedbackResponse,
    OnlineStats,
    RetrieveRequest,
    RetrieveResponse,
)

logger = logging.getLogger(__name__)


class OnlineServer:
    """HTTP server for online (test-time) adaptation.

    Key differences from offline mode:
    - No ground-truth labels; uses execution feedback only
    - Sequential processing (one request at a time)
    - Immediate adaptation after each feedback
    - No epochs; continuous learning
    """

    def __init__(
        self,
        store: Store | None = None,
        reflector: Reflector | None = None,
        retriever: Retriever | None = None,
        auto_adapt: bool = True,
    ):
        """Initialize the online server.

        Args:
            store: Playbook store (loads from config if None)
            reflector: Reflector instance (creates default if None)
            retriever: Retriever instance (creates default if None)
            auto_adapt: Whether to auto-adapt on each feedback (default True)
        """
        if store is None:
            config = load_config()
            store = Store(config.database.url)
        self.store = store
        self.reflector = reflector or Reflector()
        self.retriever = retriever or Retriever(store)
        self.auto_adapt = auto_adapt
        self.mode = AdaptationMode.ONLINE

        self.session_id = str(uuid.uuid4())[:8]
        self.stats = OnlineStats(session_id=self.session_id)

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

    def get_stats(self) -> OnlineStats:
        """Get current session statistics."""
        return self.stats

    def get_playbook_version(self) -> int:
        """Get current playbook version."""
        return self.store.load_playbook().version


def create_app(
    auto_adapt: bool = True,
    store: Store | None = None,
) -> FastAPI:
    """Create FastAPI application for online serving.

    Args:
        auto_adapt: Whether to automatically adapt on feedback
        store: Optional store instance

    Returns:
        FastAPI app instance
    """
    server_instance: list[OnlineServer] = []

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        instance = OnlineServer(
            store=store,
            auto_adapt=auto_adapt,
        )
        server_instance.append(instance)
        logger.info(
            f"Online server started: session={instance.session_id}, "
            f"auto_adapt={auto_adapt}"
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
) -> None:
    """Run the online server.

    Args:
        host: Host to bind to
        port: Port to bind to
        auto_adapt: Whether to auto-adapt on feedback
        reload: Whether to enable hot reload
    """
    import uvicorn

    logger.info(f"Starting ACE online server on {host}:{port}")
    app = create_app(auto_adapt=auto_adapt)
    uvicorn.run(app, host=host, port=port, reload=reload)
