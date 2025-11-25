# tests/test_curator.py
from datetime import datetime

from ace.core.schema import Bullet
from ace.curator import curate
from ace.reflector.schema import BulletTag, CandidateBullet, Reflection


def test_curate_empty_reflection():
    """Test that curating an empty reflection returns an empty delta."""
    reflection = Reflection()
    delta = curate(reflection)

    assert delta.ops == []


def test_curate_with_helpful_tags():
    """Test that helpful bullet tags generate INCR_HELPFUL operations."""
    reflection = Reflection(
        bullet_tags=[
            BulletTag(id="strat-00091", tag="helpful"),
            BulletTag(id="tmpl-00022", tag="helpful"),
        ]
    )
    delta = curate(reflection)

    assert len(delta.ops) == 2
    assert delta.ops[0].op == "INCR_HELPFUL"
    assert delta.ops[0].target_id == "strat-00091"
    assert delta.ops[1].op == "INCR_HELPFUL"
    assert delta.ops[1].target_id == "tmpl-00022"


def test_curate_with_harmful_tags():
    """Test that harmful bullet tags generate INCR_HARMFUL operations."""
    reflection = Reflection(
        bullet_tags=[
            BulletTag(id="trbl-00007", tag="harmful"),
            BulletTag(id="strat-00050", tag="harmful"),
        ]
    )
    delta = curate(reflection)

    assert len(delta.ops) == 2
    assert delta.ops[0].op == "INCR_HARMFUL"
    assert delta.ops[0].target_id == "trbl-00007"
    assert delta.ops[1].op == "INCR_HARMFUL"
    assert delta.ops[1].target_id == "strat-00050"


def test_curate_with_candidate_bullets():
    """Test that candidate bullets generate ADD operations."""
    reflection = Reflection(
        candidate_bullets=[
            CandidateBullet(
                section="strategies",
                content="Use hybrid retrieval for better results",
                tags=["topic:retrieval", "stack:python"],
            ),
            CandidateBullet(
                section="troubleshooting",
                content="Check FAISS index dimension mismatch",
                tags=["topic:vector", "tool:faiss"],
            ),
        ]
    )
    delta = curate(reflection)

    assert len(delta.ops) == 2

    # First ADD operation
    assert delta.ops[0].op == "ADD"
    assert delta.ops[0].new_bullet is not None
    assert delta.ops[0].new_bullet["section"] == "strategies"
    assert delta.ops[0].new_bullet["content"] == "Use hybrid retrieval for better results"
    assert delta.ops[0].new_bullet["tags"] == ["topic:retrieval", "stack:python"]

    # Second ADD operation
    assert delta.ops[1].op == "ADD"
    assert delta.ops[1].new_bullet is not None
    assert delta.ops[1].new_bullet["section"] == "troubleshooting"
    assert delta.ops[1].new_bullet["content"] == "Check FAISS index dimension mismatch"
    assert delta.ops[1].new_bullet["tags"] == ["topic:vector", "tool:faiss"]


def test_curate_with_mixed_operations():
    """Test that a reflection with both tags and candidate bullets generates
    correct mixed operations."""
    reflection = Reflection(
        bullet_tags=[
            BulletTag(id="strat-00091", tag="helpful"),
            BulletTag(id="tmpl-00022", tag="harmful"),
        ],
        candidate_bullets=[
            CandidateBullet(
                section="facts",
                content="SQLite supports JSON1 extension",
                tags=["db:sqlite", "topic:storage"],
            )
        ],
    )
    delta = curate(reflection)

    assert len(delta.ops) == 3

    # Tag operations come first
    assert delta.ops[0].op == "INCR_HELPFUL"
    assert delta.ops[0].target_id == "strat-00091"

    assert delta.ops[1].op == "INCR_HARMFUL"
    assert delta.ops[1].target_id == "tmpl-00022"

    # Then ADD operations
    assert delta.ops[2].op == "ADD"
    assert delta.ops[2].new_bullet is not None
    assert delta.ops[2].new_bullet["section"] == "facts"
    assert delta.ops[2].new_bullet["content"] == "SQLite supports JSON1 extension"


def test_curate_preserves_all_candidate_bullet_fields():
    """Test that all fields from candidate bullets are preserved in the delta."""
    reflection = Reflection(
        candidate_bullets=[
            CandidateBullet(
                section="code_snippets",
                content="import faiss\nindex = faiss.IndexFlatL2(dim)",
                tags=["lang:python", "lib:faiss", "topic:indexing"],
            )
        ]
    )
    delta = curate(reflection)

    assert len(delta.ops) == 1
    op = delta.ops[0]
    assert op.op == "ADD"
    assert op.new_bullet is not None
    assert op.new_bullet["section"] == "code_snippets"
    assert op.new_bullet["content"] == "import faiss\nindex = faiss.IndexFlatL2(dim)"
    assert len(op.new_bullet["tags"]) == 3
    assert "lang:python" in op.new_bullet["tags"]
    assert "lib:faiss" in op.new_bullet["tags"]
    assert "topic:indexing" in op.new_bullet["tags"]


def test_curate_with_reflection_insights():
    """Test that curator handles reflections with insights but focuses on actionable outputs."""
    reflection = Reflection(
        error_identification="FAISS dimension mismatch",
        root_cause_analysis="Embedding model changed from 384 to 768 dims",
        correct_approach="Rebuild index with correct dimensions",
        key_insight="Always validate embedding dimensions match index",
        candidate_bullets=[
            CandidateBullet(
                section="troubleshooting",
                content="Validate embedding dims match FAISS index dims before insertion",
                tags=["topic:vector", "tool:faiss", "error:dimension"],
            )
        ],
    )
    delta = curate(reflection)

    # Insights are captured in the Reflection but don't directly generate ops
    # Only candidate_bullets and bullet_tags generate operations
    assert len(delta.ops) == 1
    assert delta.ops[0].op == "ADD"
    assert delta.ops[0].new_bullet is not None
    assert "dims" in delta.ops[0].new_bullet["content"].lower()


# --- Semantic Duplicate Detection Tests ---


def _make_bullet(id: str, content: str, section: str = "strategies") -> Bullet:
    """Helper to create a Bullet for testing."""
    return Bullet(
        id=id,
        section=section,  # type: ignore
        content=content,
        tags=[],
        helpful=0,
        harmful=0,
        added_at=datetime.utcnow(),
    )


def test_curate_emits_patch_for_near_duplicate():
    """Test that a semantically similar candidate emits PATCH instead of ADD."""
    existing_bullets = [
        _make_bullet("strat-001", "Use hybrid retrieval with BM25 and vector search"),
    ]

    reflection = Reflection(
        candidate_bullets=[
            CandidateBullet(
                section="strategies",
                content="Use hybrid retrieval combining BM25 and embedding search",
                tags=["topic:retrieval"],
            ),
        ]
    )

    delta = curate(reflection, existing_bullets=existing_bullets)

    assert len(delta.ops) == 1
    assert delta.ops[0].op == "PATCH"
    assert delta.ops[0].target_id == "strat-001"
    assert delta.ops[0].patch == "Use hybrid retrieval combining BM25 and embedding search"


def test_curate_emits_add_for_unique_candidate():
    """Test that a unique candidate bullet emits ADD."""
    existing_bullets = [
        _make_bullet("strat-001", "Use hybrid retrieval with BM25 and vector search"),
    ]

    reflection = Reflection(
        candidate_bullets=[
            CandidateBullet(
                section="troubleshooting",
                content="Check database connection timeout settings",
                tags=["topic:database"],
            ),
        ]
    )

    delta = curate(reflection, existing_bullets=existing_bullets)

    assert len(delta.ops) == 1
    assert delta.ops[0].op == "ADD"
    assert delta.ops[0].new_bullet is not None
    assert delta.ops[0].new_bullet["content"] == "Check database connection timeout settings"


def test_curate_mixed_add_and_patch():
    """Test mixed ADD and PATCH operations when some candidates are duplicates."""
    existing_bullets = [
        _make_bullet("strat-001", "Always validate input before processing"),
        _make_bullet("trbl-002", "Log errors with full stack trace"),
    ]

    reflection = Reflection(
        candidate_bullets=[
            CandidateBullet(
                section="strategies",
                content="Validate all inputs before any processing",  # Near-dup of strat-001
                tags=["topic:validation"],
            ),
            CandidateBullet(
                section="facts",
                content="Python 3.11 supports exception groups",  # Unique
                tags=["lang:python"],
            ),
        ]
    )

    delta = curate(reflection, existing_bullets=existing_bullets)

    assert len(delta.ops) == 2
    # First should be PATCH (near-duplicate)
    assert delta.ops[0].op == "PATCH"
    assert delta.ops[0].target_id == "strat-001"
    # Second should be ADD (unique)
    assert delta.ops[1].op == "ADD"
    assert delta.ops[1].new_bullet is not None
    assert "Python 3.11" in delta.ops[1].new_bullet["content"]


def test_curate_with_empty_existing_bullets():
    """Test that all candidates become ADD when existing_bullets is empty."""
    reflection = Reflection(
        candidate_bullets=[
            CandidateBullet(
                section="strategies",
                content="Some new strategy",
                tags=["topic:new"],
            ),
        ]
    )

    delta = curate(reflection, existing_bullets=[])

    assert len(delta.ops) == 1
    assert delta.ops[0].op == "ADD"


def test_curate_legacy_behavior_without_existing_bullets():
    """Test that legacy behavior (no dedup) is preserved when existing_bullets is None."""
    reflection = Reflection(
        candidate_bullets=[
            CandidateBullet(
                section="strategies",
                content="Some strategy",
                tags=["topic:test"],
            ),
        ]
    )

    delta = curate(reflection)  # No existing_bullets

    assert len(delta.ops) == 1
    assert delta.ops[0].op == "ADD"


def test_curate_tags_processed_before_semantic_check():
    """Test that bullet_tags (INCR_HELPFUL/HARMFUL) are processed regardless of dedup."""
    existing_bullets = [
        _make_bullet(
            "strat-001", "Use hybrid retrieval with BM25 and vector search for better results"
        ),
    ]

    reflection = Reflection(
        bullet_tags=[
            BulletTag(id="strat-001", tag="helpful"),
        ],
        candidate_bullets=[
            CandidateBullet(
                section="strategies",
                content="Use hybrid retrieval combining BM25 and vector search for better results",
                tags=["topic:retrieval"],
            ),
        ],
    )

    delta = curate(reflection, existing_bullets=existing_bullets)

    assert len(delta.ops) == 2
    assert delta.ops[0].op == "INCR_HELPFUL"
    assert delta.ops[0].target_id == "strat-001"
    assert delta.ops[1].op == "PATCH"
    assert delta.ops[1].target_id == "strat-001"


def test_curate_custom_threshold():
    """Test that custom threshold affects duplicate detection."""
    existing_bullets = [
        _make_bullet("strat-001", "Use hybrid retrieval with BM25 and vector search"),
    ]

    reflection = Reflection(
        candidate_bullets=[
            CandidateBullet(
                section="strategies",
                content="Use hybrid retrieval",  # Less similar
                tags=["topic:retrieval"],
            ),
        ]
    )

    # With very high threshold, should be ADD (not similar enough)
    delta_high = curate(reflection, existing_bullets=existing_bullets, threshold=0.99)
    assert delta_high.ops[0].op == "ADD"

    # With lower threshold, should be PATCH
    delta_low = curate(reflection, existing_bullets=existing_bullets, threshold=0.50)
    assert delta_low.ops[0].op == "PATCH"
