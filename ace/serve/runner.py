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
from html import escape
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from ace.core.config import ACEConfig, load_config
from ace.core.logging_utils import configure_logging
from ace.core.merge import Delta as MergeDelta
from ace.core.merge import apply_delta
from ace.core.metrics import MetricsTracker, get_tracker
from ace.core.retrieve import Retriever
from ace.core.schema import Bullet, Playbook, Section
from ace.core.storage.store_adapter import Store
from ace.curator.curator import curate
from ace.generator.schemas import TrajectoryDoc
from ace.refine.runner import refine as run_refine
from ace.reflector.reflector import Reflector
from ace.reflector.schema import BulletTag, CandidateBullet, Reflection

from .prometheus import build_metrics_registry, metrics_response
from .schema import (
    AdaptationMode,
    CommitRequest,
    CurateRequest,
    FeedbackRequest,
    FeedbackResponse,
    OnlineStats,
    RefineRequest,
    ReflectRequest,
    RetrieveRequest,
    RetrieveResponse,
    WarmupSource,
)

logger = logging.getLogger(__name__)

SECTION_ORDER: tuple[Section, ...] = (
    "strategies_and_hard_rules",
    "code_snippets_and_templates",
    "troubleshooting_and_pitfalls",
    "domain_facts_and_references",
)

SECTION_LABELS: dict[Section, str] = {
    "strategies_and_hard_rules": "Strategies and Hard Rules",
    "code_snippets_and_templates": "Code Snippets and Templates",
    "troubleshooting_and_pitfalls": "Troubleshooting and Pitfalls",
    "domain_facts_and_references": "Domain Facts and References",
}

SECTION_DESCRIPTIONS: dict[Section, str] = {
    "strategies_and_hard_rules": "Operating heuristics, policies, and durable tactics.",
    "code_snippets_and_templates": "Reusable implementation patterns and scaffolds.",
    "troubleshooting_and_pitfalls": "Failure signatures, debugging cues, and avoidance rules.",
    "domain_facts_and_references": "Reference material and stable domain knowledge.",
}


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

    def _load_warmup_playbook(self, warmup_path: str | Path) -> tuple[WarmupSource, int, int]:
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
            doc = TrajectoryDoc(
                query=request.query,
                retrieved_bullet_ids=request.retrieved_bullet_ids,
                code_diff=request.code_diff,
                test_output=request.test_output,
                logs=request.logs,
                env_meta=request.env_meta or {},
            )
            reflection = self.reflector.reflect(doc)

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

    @staticmethod
    def _serialize_reflection(reflection: Reflection) -> dict[str, Any]:
        """Convert a reflection dataclass into the public JSON shape."""
        return {
            "error_identification": reflection.error_identification,
            "root_cause_analysis": reflection.root_cause_analysis,
            "correct_approach": reflection.correct_approach,
            "key_insight": reflection.key_insight,
            "bullet_tags": [
                {"id": bullet_tag.id, "tag": bullet_tag.tag}
                for bullet_tag in reflection.bullet_tags
            ],
            "candidate_bullets": [
                {
                    "section": candidate.section,
                    "content": candidate.content,
                    "tags": candidate.tags,
                }
                for candidate in reflection.candidate_bullets
            ],
            "iteration": reflection.iteration,
            "parent_id": reflection.parent_id,
        }

    @staticmethod
    def _build_reflection(reflection_data: dict[str, Any]) -> Reflection:
        """Convert REST JSON payloads into the reflection dataclass."""
        return Reflection(
            error_identification=reflection_data.get("error_identification"),
            root_cause_analysis=reflection_data.get("root_cause_analysis"),
            correct_approach=reflection_data.get("correct_approach"),
            key_insight=reflection_data.get("key_insight"),
            bullet_tags=[
                BulletTag(id=tag["id"], tag=tag["tag"])
                for tag in reflection_data.get("bullet_tags", [])
            ],
            candidate_bullets=[
                CandidateBullet(
                    section=bullet["section"],
                    content=bullet["content"],
                    tags=bullet.get("tags", []),
                )
                for bullet in reflection_data.get("candidate_bullets", [])
            ],
            iteration=reflection_data.get("iteration", 0),
            parent_id=reflection_data.get("parent_id"),
        )

    def _persist_refined_playbook(
        self,
        playbook: Playbook,
        original_ids: set[str],
    ) -> int:
        """Persist bullet removals and survivors after refine mutates the playbook."""
        refined_ids = {bullet.id for bullet in playbook.bullets}
        removed_ids = original_ids - refined_ids

        for bullet_id in removed_ids:
            self.store.delete_bullet(bullet_id)

        for bullet in playbook.bullets:
            self.store.save_bullet(bullet)

        return len(removed_ids)

    def reflect(self, doc_data: dict[str, Any]) -> dict[str, Any]:
        """Generate a reflection from a trajectory document."""
        doc = TrajectoryDoc(**doc_data)
        reflection = self.reflector.reflect(doc)
        return self._serialize_reflection(reflection)

    def curate(self, reflection_data: dict[str, Any]) -> dict[str, Any]:
        """Convert a reflection payload into delta operations."""
        reflection = self._build_reflection(reflection_data)
        playbook = self.store.load_playbook()
        delta = curate(reflection, existing_bullets=playbook.bullets)
        return delta.model_dump()

    def commit(self, delta_data: dict[str, Any]) -> dict[str, int]:
        """Apply a delta to the current playbook."""
        playbook = self.store.load_playbook()
        delta = MergeDelta.from_dict(delta_data)
        new_playbook = apply_delta(playbook, delta, self.store)
        return {"version": new_playbook.version}

    def refine(self, threshold: float = 0.90) -> dict[str, int]:
        """Run manual playbook refinement and persist the updated playbook."""
        playbook = self.store.load_playbook()
        original_ids = {bullet.id for bullet in playbook.bullets}
        result = run_refine(Reflection(), playbook, threshold=threshold)
        self._persist_refined_playbook(playbook, original_ids)
        return {"merged": result.merged, "archived": result.archived}

    def get_playbook(self) -> dict[str, Any]:
        """Return the current playbook as JSON-serializable data."""
        return self.store.load_playbook().model_dump()

    def render_playbook_view(self) -> str:
        """Render the current playbook as a lightweight HTML dashboard."""
        playbook = self.store.load_playbook()
        bullets_by_section: dict[Section, list[Bullet]] = {section: [] for section in SECTION_ORDER}
        all_tags: list[str] = []
        helpful_total = 0
        harmful_total = 0
        recently_used = 0

        for bullet in playbook.bullets:
            bullets_by_section.setdefault(bullet.section, []).append(bullet)
            all_tags.extend(bullet.tags)
            helpful_total += bullet.helpful
            harmful_total += bullet.harmful
            if bullet.last_used is not None:
                recently_used += 1

        def section_sort_key(item: Bullet) -> tuple[int, int, str]:
            return (-item.helpful, item.harmful, item.id)

        section_markup: list[str] = []
        total_bullets = len(playbook.bullets)
        top_tags = sorted({tag for tag in all_tags})[:12]

        for section in SECTION_ORDER:
            bullets = sorted(bullets_by_section.get(section, []), key=section_sort_key)
            cards = "".join(self._render_bullet_card(bullet) for bullet in bullets)
            empty_state = (
                ""
                if bullets
                else "<div class='empty-state'>No bullets stored in this section yet.</div>"
            )
            section_markup.append(
                f"""
                <section class="section-panel" data-section="{escape(section)}">
                  <div class="section-header">
                    <div>
                      <p class="section-kicker">{escape(section.replace('_', ' '))}</p>
                      <h2>{escape(SECTION_LABELS[section])}</h2>
                    </div>
                    <span class="section-count">{len(bullets)} bullets</span>
                  </div>
                  <p class="section-description">{escape(SECTION_DESCRIPTIONS[section])}</p>
                  <div class="bullet-grid">
                    {cards or empty_state}
                  </div>
                </section>
                """
            )

        tag_markup = "".join(
            f"<span class='tag-chip'>{escape(tag)}</span>"
            for tag in top_tags
        ) or "<span class='tag-chip muted'>No tags yet</span>"

        return f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>ACE Playbook Viewer</title>
    <style>
      :root {{
        --bg: #f4efe2;
        --surface: rgba(255, 252, 245, 0.86);
        --surface-strong: #fff9ef;
        --text: #1f2a1f;
        --muted: #5d6758;
        --line: rgba(31, 42, 31, 0.12);
        --accent: #1e6b52;
        --accent-soft: rgba(30, 107, 82, 0.12);
        --warn: #8a4b1f;
        --shadow: 0 20px 40px rgba(31, 42, 31, 0.10);
      }}

      * {{ box-sizing: border-box; }}

      body {{
        margin: 0;
        font-family: "Avenir Next", "Segoe UI", sans-serif;
        color: var(--text);
        background:
          radial-gradient(circle at top left, rgba(30, 107, 82, 0.18), transparent 32%),
          radial-gradient(circle at top right, rgba(138, 75, 31, 0.15), transparent 28%),
          linear-gradient(180deg, #f9f4e8 0%, var(--bg) 42%, #efe5d0 100%);
      }}

      main {{
        width: min(1180px, calc(100vw - 32px));
        margin: 0 auto;
        padding: 32px 0 48px;
      }}

      .hero {{
        background: linear-gradient(135deg, rgba(255, 249, 239, 0.96), rgba(255, 252, 245, 0.72));
        border: 1px solid var(--line);
        border-radius: 28px;
        box-shadow: var(--shadow);
        overflow: hidden;
      }}

      .hero-inner {{
        display: grid;
        gap: 24px;
        grid-template-columns: 2fr 1fr;
        padding: 32px;
      }}

      .eyebrow {{
        display: inline-flex;
        padding: 8px 12px;
        border-radius: 999px;
        background: var(--accent-soft);
        color: var(--accent);
        font-size: 0.82rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }}

      h1, h2, h3, .stat-number {{
        font-family: "Iowan Old Style", "Palatino Linotype", serif;
      }}

      h1 {{
        margin: 14px 0 12px;
        font-size: clamp(2.4rem, 4vw, 4.4rem);
        line-height: 0.98;
      }}

      .hero-copy p,
      .section-description,
      .bullet-meta,
      .search-note {{
        color: var(--muted);
      }}

      .hero-copy p {{
        margin: 0;
        max-width: 62ch;
        font-size: 1rem;
        line-height: 1.6;
      }}

      .stats-grid {{
        display: grid;
        gap: 12px;
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}

      .stat-card {{
        padding: 18px;
        border-radius: 20px;
        background: var(--surface);
        border: 1px solid var(--line);
      }}

      .stat-label {{
        display: block;
        color: var(--muted);
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }}

      .stat-number {{
        display: block;
        margin-top: 8px;
        font-size: 2rem;
      }}

      .controls {{
        margin-top: 20px;
        display: grid;
        gap: 16px;
        grid-template-columns: 1.2fr 1fr;
        align-items: start;
      }}

      .search-panel,
      .tag-panel,
      .section-panel {{
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 24px;
        box-shadow: var(--shadow);
      }}

      .search-panel,
      .tag-panel {{
        padding: 20px;
      }}

      .search-label {{
        display: block;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--muted);
      }}

      .search-input {{
        width: 100%;
        margin-top: 10px;
        padding: 14px 16px;
        border-radius: 16px;
        border: 1px solid rgba(31, 42, 31, 0.16);
        background: var(--surface-strong);
        color: var(--text);
        font-size: 1rem;
      }}

      .search-input:focus {{
        outline: 2px solid rgba(30, 107, 82, 0.22);
        border-color: var(--accent);
      }}

      .search-note {{
        margin: 10px 0 0;
        font-size: 0.92rem;
      }}

      .tag-panel h3 {{
        margin: 0 0 12px;
        font-size: 1.1rem;
      }}

      .tag-row {{
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
      }}

      .tag-chip {{
        display: inline-flex;
        padding: 8px 12px;
        border-radius: 999px;
        background: rgba(31, 42, 31, 0.06);
        color: var(--text);
        font-size: 0.92rem;
      }}

      .tag-chip.muted {{
        color: var(--muted);
      }}

      .sections {{
        margin-top: 24px;
        display: grid;
        gap: 18px;
      }}

      .section-panel {{
        padding: 24px;
      }}

      .section-header {{
        display: flex;
        gap: 16px;
        justify-content: space-between;
        align-items: baseline;
      }}

      .section-kicker {{
        margin: 0 0 8px;
        color: var(--accent);
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
      }}

      .section-header h2 {{
        margin: 0;
        font-size: 1.8rem;
      }}

      .section-count {{
        padding: 6px 12px;
        border-radius: 999px;
        background: rgba(31, 42, 31, 0.06);
        color: var(--muted);
        white-space: nowrap;
      }}

      .section-description {{
        margin: 10px 0 0;
        line-height: 1.5;
      }}

      .bullet-grid {{
        margin-top: 20px;
        display: grid;
        gap: 14px;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      }}

      .bullet-card {{
        padding: 18px;
        border-radius: 20px;
        background: var(--surface-strong);
        border: 1px solid rgba(31, 42, 31, 0.08);
      }}

      .bullet-id {{
        font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
        font-size: 0.85rem;
        color: var(--accent);
      }}

      .bullet-content {{
        margin: 12px 0 14px;
        font-size: 1rem;
        line-height: 1.55;
      }}

      .bullet-meta {{
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        font-size: 0.88rem;
      }}

      .bullet-tags {{
        margin-top: 12px;
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
      }}

      .bullet-tag {{
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(30, 107, 82, 0.10);
        color: var(--accent);
        font-size: 0.82rem;
      }}

      .empty-state {{
        padding: 24px;
        border-radius: 18px;
        border: 1px dashed rgba(31, 42, 31, 0.18);
        color: var(--muted);
        text-align: center;
      }}

      .is-hidden {{
        display: none !important;
      }}

      @media (max-width: 860px) {{
        .hero-inner,
        .controls {{
          grid-template-columns: 1fr;
        }}

        .stats-grid {{
          grid-template-columns: repeat(2, minmax(0, 1fr));
        }}
      }}

      @media (max-width: 560px) {{
        main {{
          width: min(100vw - 20px, 1180px);
          padding-top: 20px;
        }}

        .hero-inner,
        .section-panel,
        .search-panel,
        .tag-panel {{
          padding: 18px;
        }}

        .stats-grid {{
          grid-template-columns: 1fr;
        }}

        .section-header {{
          flex-direction: column;
          align-items: flex-start;
        }}
      }}
    </style>
  </head>
  <body>
    <main>
      <section class="hero">
        <div class="hero-inner">
          <div class="hero-copy">
            <span class="eyebrow">ACE Playbook Viewer</span>
            <h1>Inspect the live playbook without losing the JSON contract.</h1>
            <p>
              This view sits on top of the same store that powers retrieval and
              adaptation. Use it to audit section balance, bullet quality, and tag
              coverage before you refine or patch the playbook.
            </p>
          </div>
          <div class="stats-grid">
            <div class="stat-card">
              <span class="stat-label">Version</span>
              <span class="stat-number">{playbook.version}</span>
            </div>
            <div class="stat-card">
              <span class="stat-label">Bullets</span>
              <span class="stat-number">{total_bullets}</span>
            </div>
            <div class="stat-card">
              <span class="stat-label">Helpful Marks</span>
              <span class="stat-number">{helpful_total}</span>
            </div>
            <div class="stat-card">
              <span class="stat-label">Used at Least Once</span>
              <span class="stat-number">{recently_used}</span>
            </div>
          </div>
        </div>
      </section>

      <section class="controls">
        <div class="search-panel">
          <label class="search-label" for="playbook-search">Filter bullets</label>
          <input
            class="search-input"
            id="playbook-search"
            type="search"
            placeholder="Search by id, content, section, or tag"
          />
          <p class="search-note">
            Harmful marks tracked: <strong>{harmful_total}</strong>.
            Filtering happens in-browser on the live page.
          </p>
        </div>
        <div class="tag-panel">
          <h3>Observed tags</h3>
          <div class="tag-row">{tag_markup}</div>
        </div>
      </section>

      <section class="sections" id="playbook-sections">
        {''.join(section_markup)}
      </section>
    </main>

    <script>
      const searchInput = document.getElementById("playbook-search");
      const cards = Array.from(document.querySelectorAll(".bullet-card"));
      const sections = Array.from(document.querySelectorAll(".section-panel"));

      function updateVisibility() {{
        const query = searchInput.value.trim().toLowerCase();

        for (const card of cards) {{
          const haystack = card.dataset.search || "";
          card.classList.toggle("is-hidden", query.length > 0 && !haystack.includes(query));
        }}

        for (const section of sections) {{
          const visibleCards = section.querySelectorAll(".bullet-card:not(.is-hidden)");
          const emptyState = section.querySelector(".empty-state");
          section.classList.toggle("is-hidden", query.length > 0 && visibleCards.length === 0);
          if (emptyState) {{
            emptyState.classList.toggle("is-hidden", query.length > 0);
          }}
        }}
      }}

      searchInput.addEventListener("input", updateVisibility);
    </script>
  </body>
</html>
"""

    @staticmethod
    def _render_bullet_card(bullet: Bullet) -> str:
        """Render a single bullet card for the HTML playbook view."""
        tags = "".join(
            f"<span class='bullet-tag'>{escape(tag)}</span>"
            for tag in bullet.tags
        ) or "<span class='bullet-tag'>untagged</span>"
        search_text = " ".join([bullet.id, bullet.section, bullet.content, *bullet.tags]).lower()
        last_used = bullet.last_used.isoformat() if bullet.last_used is not None else "Never"
        added_at = bullet.added_at.isoformat()
        return f"""
        <article class="bullet-card" data-search="{escape(search_text, quote=True)}">
          <div class="bullet-id">{escape(bullet.id)}</div>
          <p class="bullet-content">{escape(bullet.content)}</p>
          <div class="bullet-meta">
            <span>Helpful {bullet.helpful}</span>
            <span>Harmful {bullet.harmful}</span>
            <span>Added {escape(added_at)}</span>
            <span>Last used {escape(last_used)}</span>
          </div>
          <div class="bullet-tags">{tags}</div>
        </article>
        """

    def _update_avg_adaptation_ms(self, new_ms: float) -> None:
        """Update running average of adaptation time."""
        n = self.stats.requests_processed
        if n == 1:
            self.stats.avg_adaptation_ms = new_ms
        else:
            self.stats.avg_adaptation_ms = (self.stats.avg_adaptation_ms * (n - 1) + new_ms) / n

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
            logger.info(f"Auto-refine triggered: {bullet_count} bullets > max {self.max_bullets}")

        if not should_refine:
            return

        original_ids = {b.id for b in playbook.bullets}

        empty_reflection = Reflection()
        result = run_refine(
            empty_reflection,
            playbook,
            threshold=self._config.refine.threshold,
        )

        removed_count = self._persist_refined_playbook(playbook, original_ids)

        self._delta_count_since_refine = 0

        self.stats.auto_refine_runs += 1
        self.stats.auto_refine_merged += result.merged
        self.stats.auto_refine_archived += result.archived

        logger.info(
            f"Auto-refine complete: merged={result.merged}, archived={result.archived}, "
            f"removed={removed_count}, bullets={len(playbook.bullets)}"
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
    metrics_tracker: MetricsTracker | None = None,
) -> FastAPI:
    """Create FastAPI application for online serving.

    Args:
        auto_adapt: Whether to automatically adapt on feedback
        store: Optional store instance
        warmup_path: Path to playbook JSON file for warm-start
        auto_refine_every: Run refine every N deltas (0 = disabled)
        max_bullets: Max bullets before triggering refine (overrides config)
        metrics_tracker: Optional validation metrics tracker override

    Returns:
        FastAPI app instance
    """
    server_instance: list[OnlineServer] = []
    tracker = metrics_tracker or get_tracker()

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

    metrics_registry = build_metrics_registry(get_server, tracker)

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

    @app.post("/reflect")
    async def reflect(request: ReflectRequest) -> dict[str, Any]:
        """Generate a reflection from a trajectory document."""
        return get_server().reflect(request.doc)

    @app.post("/curate")
    async def curate_playbook(request: CurateRequest) -> dict[str, Any]:
        """Convert a reflection into delta operations."""
        return get_server().curate(request.reflection)

    @app.post("/commit")
    async def commit(request: CommitRequest) -> dict[str, int]:
        """Apply a delta to the playbook."""
        return get_server().commit(request.delta)

    @app.post("/refine")
    async def refine(request: RefineRequest) -> dict[str, int]:
        """Run playbook refinement."""
        return get_server().refine(request.threshold)

    @app.get("/stats")
    async def stats() -> dict[str, Any]:
        """Get session statistics."""
        return get_server().get_stats().model_dump()

    @app.get("/metrics")
    async def metrics():
        """Expose Prometheus metrics for the online ACE server."""
        return metrics_response(metrics_registry)

    @app.get("/playbook")
    async def playbook() -> dict[str, Any]:
        """Get the full playbook."""
        return get_server().get_playbook()

    @app.get("/playbook/view", response_class=HTMLResponse)
    async def playbook_view() -> HTMLResponse:
        """Render the playbook as a human-friendly HTML dashboard."""
        return HTMLResponse(get_server().render_playbook_view())

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

    config = load_config()
    configure_logging(config.logging.level, config.logging.format)

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
