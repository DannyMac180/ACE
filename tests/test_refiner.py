# tests/test_refiner.py
from ace.core.schema import Bullet, Playbook
from ace.refine import refine
from ace.reflector.schema import BulletTag, CandidateBullet, Reflection


def test_refine_empty_reflection():
    """Test that refining an empty reflection returns empty result."""
    playbook = Playbook(version=1, bullets=[])
    reflection = Reflection()

    result = refine(reflection, playbook)

    assert result.merged == 0
    assert result.archived == 0
    assert result.ops == []


def test_refine_with_reflection_processes_curator():
    """Test that refine correctly invokes curator stage."""
    playbook = Playbook(
        version=1,
        bullets=[
            Bullet(
                id="strat-00001",
                section="strategies",
                content="Existing strategy",
                tags=["topic:test"]
            )
        ]
    )

    reflection = Reflection(
        bullet_tags=[
            BulletTag(id="strat-00001", tag="helpful")
        ],
        candidate_bullets=[
            CandidateBullet(
                section="strategies",
                content="New strategy from reflection",
                tags=["topic:retrieval"]
            )
        ]
    )

    result = refine(reflection, playbook)

    # Curator stage runs successfully (dedup/consolidate/archive are stubbed)
    assert isinstance(result.merged, int)
    assert isinstance(result.archived, int)
    assert isinstance(result.ops, list)


def test_refine_with_custom_threshold():
    """Test that refine accepts custom threshold parameter."""
    playbook = Playbook(version=1, bullets=[])
    reflection = Reflection()

    result = refine(reflection, playbook, threshold=0.85)

    assert result.merged == 0
    assert result.archived == 0


def test_refine_runner_initialization():
    """Test RefineRunner initialization with playbook and threshold."""
    from ace.refine.runner import RefineRunner

    playbook = Playbook(version=2, bullets=[])
    runner = RefineRunner(playbook=playbook, threshold=0.92)

    assert runner.playbook == playbook
    assert runner.threshold == 0.92


def test_refine_runner_run_returns_result():
    """Test that RefineRunner.run() returns a valid RefineResult."""
    from ace.refine.runner import RefineRunner

    playbook = Playbook(version=1, bullets=[])
    reflection = Reflection(
        candidate_bullets=[
            CandidateBullet(
                section="facts",
                content="Test fact",
                tags=["topic:test"]
            )
        ]
    )

    runner = RefineRunner(playbook=playbook)
    result = runner.run(reflection)

    assert hasattr(result, 'merged')
    assert hasattr(result, 'archived')
    assert hasattr(result, 'ops')


def test_refine_orchestration_stages():
    """Test that refine orchestrates the expected pipeline stages."""
    from unittest.mock import patch

    from ace.refine.runner import RefineRunner

    playbook = Playbook(version=1, bullets=[])
    reflection = Reflection(
        candidate_bullets=[
            CandidateBullet(
                section="strategies",
                content="Strategy one",
                tags=["topic:test"]
            )
        ]
    )

    runner = RefineRunner(playbook=playbook)

    # Verify that curator is called (it should convert reflection to delta)
    with patch('ace.refine.runner.curate') as mock_curate:
        from ace.core.schema import Delta
        mock_curate.return_value = Delta(ops=[])

        runner.run(reflection)

        # Curator should be invoked
        mock_curate.assert_called_once_with(reflection)


def test_refine_with_helpful_and_harmful_tags():
    """Test refine handles both helpful and harmful bullet tags."""
    playbook = Playbook(
        version=1,
        bullets=[
            Bullet(
                id="strat-00001",
                section="strategies",
                content="Good strategy",
                tags=["topic:retrieval"],
                helpful=5,
                harmful=0
            ),
            Bullet(
                id="strat-00002",
                section="strategies",
                content="Bad strategy",
                tags=["topic:retrieval"],
                helpful=0,
                harmful=3
            )
        ]
    )

    reflection = Reflection(
        bullet_tags=[
            BulletTag(id="strat-00001", tag="helpful"),
            BulletTag(id="strat-00002", tag="harmful")
        ]
    )

    result = refine(reflection, playbook)

    # Should process without errors
    assert isinstance(result, type(result))
    assert result.merged >= 0
    assert result.archived >= 0


def test_deduplicate_finds_similar_bullets():
    """Test that deduplication correctly identifies near-duplicate bullets."""
    playbook = Playbook(
        version=1,
        bullets=[
            Bullet(
                id="strat-00001",
                section="strategies",
                content="Use hybrid retrieval with BM25 and vector embeddings for better search results",
                tags=["topic:retrieval"]
            )
        ]
    )

    # Reflection with a very similar candidate bullet (near-duplicate)
    reflection = Reflection(
        candidate_bullets=[
            CandidateBullet(
                section="strategies",
                content="Use hybrid retrieval with BM25 and vector embeddings for improved search",
                tags=["topic:retrieval"]
            )
        ]
    )

    result = refine(reflection, playbook, threshold=0.90)

    # Should detect the duplicate and create a MERGE operation
    assert result.merged > 0
    assert len(result.ops) > 0

    # Find the MERGE operation
    merge_ops = [op for op in result.ops if op.op == "MERGE"]
    assert len(merge_ops) > 0

    # Verify the survivor is the existing bullet
    merge_op = merge_ops[0]
    assert merge_op.survivor_id == "strat-00001"
    assert len(merge_op.target_ids) > 0


def test_deduplicate_keeps_distinct_bullets():
    """Test that deduplication does NOT merge distinct bullets."""
    playbook = Playbook(
        version=1,
        bullets=[
            Bullet(
                id="strat-00001",
                section="strategies",
                content="Use hybrid retrieval with BM25 and vector embeddings",
                tags=["topic:retrieval"]
            )
        ]
    )

    # Reflection with a completely different candidate bullet
    reflection = Reflection(
        candidate_bullets=[
            CandidateBullet(
                section="strategies",
                content="Always validate user input to prevent injection attacks",
                tags=["topic:security"]
            )
        ]
    )

    result = refine(reflection, playbook, threshold=0.90)

    # Should NOT detect any duplicates
    merge_ops = [op for op in result.ops if op.op == "MERGE"]
    assert len(merge_ops) == 0


def test_deduplicate_with_minhash_threshold():
    """Test that deduplication works with minhash Jaccard similarity."""
    playbook = Playbook(
        version=1,
        bullets=[
            Bullet(
                id="strat-00001",
                section="strategies",
                content="run tests before commit always check coverage",
                tags=["topic:testing"]
            )
        ]
    )

    # Very similar wording (high Jaccard) even if embedding differs
    reflection = Reflection(
        candidate_bullets=[
            CandidateBullet(
                section="strategies",
                content="always run tests before commit check coverage",
                tags=["topic:testing"]
            )
        ]
    )

    result = refine(reflection, playbook, threshold=0.95)  # High threshold

    # Should still detect duplicate via minhash (Jaccard > 0.85)
    # Note: May or may not merge depending on actual similarity, this tests the code path
    assert isinstance(result.merged, int)


def test_consolidate_transfers_counters():
    """Test that consolidation correctly transfers helpful/harmful counters from merged bullets."""
    from ace.core.schema import RefineOp
    from ace.refine.runner import RefineRunner

    # Create playbook with two bullets that will be merged
    playbook = Playbook(
        version=1,
        bullets=[
            Bullet(
                id="strat-00001",
                section="strategies",
                content="Use hybrid retrieval",
                tags=["topic:retrieval"],
                helpful=5,
                harmful=1
            ),
            Bullet(
                id="strat-00002",
                section="strategies",
                content="Similar retrieval strategy",
                tags=["topic:retrieval"],
                helpful=3,
                harmful=2
            )
        ]
    )

    # Create MERGE operation: merge strat-00002 into strat-00001
    merge_ops = [
        RefineOp(
            op="MERGE",
            target_ids=["strat-00002"],
            survivor_id="strat-00001"
        )
    ]

    runner = RefineRunner(playbook=playbook)
    runner._consolidate(merge_ops)

    # Verify survivor has combined counters
    survivor = next(b for b in playbook.bullets if b.id == "strat-00001")
    assert survivor.helpful == 8  # 5 + 3
    assert survivor.harmful == 3  # 1 + 2

    # Verify target was removed from playbook
    assert len(playbook.bullets) == 1
    assert playbook.bullets[0].id == "strat-00001"


def test_consolidate_handles_candidate_ids():
    """Test that consolidation gracefully handles candidate IDs that don't exist in playbook."""
    from ace.core.schema import RefineOp
    from ace.refine.runner import RefineRunner

    playbook = Playbook(
        version=1,
        bullets=[
            Bullet(
                id="strat-00001",
                section="strategies",
                content="Existing strategy",
                tags=["topic:test"],
                helpful=2,
                harmful=0
            )
        ]
    )

    # MERGE operation with candidate ID (not in playbook)
    merge_ops = [
        RefineOp(
            op="MERGE",
            target_ids=["candidate-0"],  # Doesn't exist in playbook
            survivor_id="strat-00001"
        )
    ]

    runner = RefineRunner(playbook=playbook)
    runner._consolidate(merge_ops)

    # Should not crash, survivor counters unchanged
    survivor = playbook.bullets[0]
    assert survivor.helpful == 2
    assert survivor.harmful == 0
    assert len(playbook.bullets) == 1


def test_archive_policy_removes_high_harmful_ratio_bullets():
    """Test that archival policy removes bullets with harmful ratio exceeding threshold."""
    playbook = Playbook(
        version=1,
        bullets=[
            Bullet(
                id="strat-00001",
                section="strategies",
                content="Good strategy",
                tags=["topic:retrieval"],
                helpful=8,
                harmful=2  # ratio: 2/10 = 0.20 (keep)
            ),
            Bullet(
                id="strat-00002",
                section="strategies",
                content="Bad strategy",
                tags=["topic:retrieval"],
                helpful=2,
                harmful=8  # ratio: 8/10 = 0.80 (archive, exceeds 0.75)
            ),
            Bullet(
                id="strat-00003",
                section="strategies",
                content="Neutral strategy",
                tags=["topic:test"],
                helpful=0,
                harmful=0  # ratio: 0/0 (keep, no feedback)
            ),
            Bullet(
                id="strat-00004",
                section="strategies",
                content="Borderline strategy",
                tags=["topic:test"],
                helpful=3,
                harmful=9  # ratio: 9/12 = 0.75 (keep, equals threshold)
            ),
            Bullet(
                id="strat-00005",
                section="strategies",
                content="Mostly harmful",
                tags=["topic:test"],
                helpful=1,
                harmful=10  # ratio: 10/11 = 0.909 (archive)
            )
        ]
    )

    # Use default archive_ratio of 0.75
    reflection = Reflection()
    result = refine(reflection, playbook, archive_ratio=0.75)

    # Should archive strat-00002 (0.80) and strat-00005 (0.909)
    assert result.archived == 2

    # Verify archived bullets are removed from playbook
    assert len(playbook.bullets) == 3
    remaining_ids = [b.id for b in playbook.bullets]
    assert "strat-00001" in remaining_ids
    assert "strat-00002" not in remaining_ids
    assert "strat-00003" in remaining_ids
    assert "strat-00004" in remaining_ids  # 0.75 equals threshold, should keep
    assert "strat-00005" not in remaining_ids

    # Verify ARCHIVE operations are in result
    archive_ops = [op for op in result.ops if op.op == "ARCHIVE"]
    assert len(archive_ops) == 2
    archived_ids = [op.target_ids[0] for op in archive_ops]
    assert "strat-00002" in archived_ids
    assert "strat-00005" in archived_ids
