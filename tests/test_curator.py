# tests/test_curator.py
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
