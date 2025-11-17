import pytest

from ace.core.manager import PlaybookManager
from ace.core.schema import DeltaOp


def test_add_operation():
    manager = PlaybookManager()
    assert manager.playbook.version == 0
    assert len(manager.playbook.bullets) == 0

    delta = DeltaOp(
        op="ADD",
        new_bullet={
            "section": "strategies",
            "content": "Use hybrid retrieval",
            "tags": ["topic:retrieval"],
        },
    )
    manager.apply_delta(delta)

    assert manager.playbook.version == 1
    assert len(manager.playbook.bullets) == 1
    assert manager.playbook.bullets[0].content == "Use hybrid retrieval"
    assert manager.playbook.bullets[0].section == "strategies"
    assert manager.playbook.bullets[0].tags == ["topic:retrieval"]
    assert manager.playbook.bullets[0].helpful == 0
    assert manager.playbook.bullets[0].harmful == 0


def test_add_without_new_bullet_raises():
    manager = PlaybookManager()
    delta = DeltaOp(op="ADD")

    with pytest.raises(ValueError, match="'ADD' operation requires new_bullet"):
        manager.apply_delta(delta)


def test_patch_operation():
    manager = PlaybookManager()

    delta_add = DeltaOp(
        op="ADD",
        new_bullet={"section": "strategies", "content": "Original content", "tags": []},
    )
    manager.apply_delta(delta_add)
    bullet_id = manager.playbook.bullets[0].id

    delta_patch = DeltaOp(op="PATCH", target_id=bullet_id, patch="Updated content")
    manager.apply_delta(delta_patch)

    assert manager.playbook.version == 2
    assert manager.playbook.bullets[0].content == "Updated content"


def test_patch_without_target_raises():
    manager = PlaybookManager()
    delta = DeltaOp(op="PATCH", patch="New content")

    with pytest.raises(ValueError, match="'PATCH' operation requires target_id and patch"):
        manager.apply_delta(delta)


def test_patch_nonexistent_bullet_raises():
    manager = PlaybookManager()
    delta = DeltaOp(op="PATCH", target_id="nonexistent", patch="Content")

    with pytest.raises(ValueError, match="Bullet not found"):
        manager.apply_delta(delta)


def test_incr_helpful_operation():
    manager = PlaybookManager()

    delta_add = DeltaOp(
        op="ADD", new_bullet={"section": "strategies", "content": "Test", "tags": []}
    )
    manager.apply_delta(delta_add)
    bullet_id = manager.playbook.bullets[0].id

    delta_incr = DeltaOp(op="INCR_HELPFUL", target_id=bullet_id)
    manager.apply_delta(delta_incr)

    assert manager.playbook.version == 2
    assert manager.playbook.bullets[0].helpful == 1
    assert manager.playbook.bullets[0].last_used is not None


def test_incr_harmful_operation():
    manager = PlaybookManager()

    delta_add = DeltaOp(
        op="ADD", new_bullet={"section": "strategies", "content": "Test", "tags": []}
    )
    manager.apply_delta(delta_add)
    bullet_id = manager.playbook.bullets[0].id

    delta_incr = DeltaOp(op="INCR_HARMFUL", target_id=bullet_id)
    manager.apply_delta(delta_incr)

    assert manager.playbook.version == 2
    assert manager.playbook.bullets[0].harmful == 1


def test_deprecate_operation():
    manager = PlaybookManager()

    delta_add = DeltaOp(
        op="ADD", new_bullet={"section": "strategies", "content": "Test", "tags": []}
    )
    manager.apply_delta(delta_add)
    bullet_id = manager.playbook.bullets[0].id

    assert len(manager.playbook.bullets) == 1

    delta_deprecate = DeltaOp(op="DEPRECATE", target_id=bullet_id)
    manager.apply_delta(delta_deprecate)

    assert manager.playbook.version == 2
    assert len(manager.playbook.bullets) == 0


def test_invalid_operation_raises():
    manager = PlaybookManager()
    delta = DeltaOp(op="INVALID_OP")

    with pytest.raises(ValueError, match="Invalid operation: INVALID_OP"):
        manager.apply_delta(delta)


def test_multiple_operations_idempotent():
    manager = PlaybookManager()

    delta1 = DeltaOp(
        op="ADD", new_bullet={"section": "strategies", "content": "Bullet 1", "tags": []}
    )
    delta2 = DeltaOp(
        op="ADD", new_bullet={"section": "facts", "content": "Bullet 2", "tags": ["tag1"]}
    )

    manager.apply_delta(delta1)
    manager.apply_delta(delta2)

    assert manager.playbook.version == 2
    assert len(manager.playbook.bullets) == 2

    bullet_id = manager.playbook.bullets[0].id
    delta3 = DeltaOp(op="INCR_HELPFUL", target_id=bullet_id)
    manager.apply_delta(delta3)

    assert manager.playbook.version == 3
    assert manager.playbook.bullets[0].helpful == 1


def test_add_with_id_idempotency():
    """ADD with explicit ID is idempotent: replaying same ADD is a no-op."""
    manager = PlaybookManager()
    delta_add = DeltaOp(
        op="ADD",
        new_bullet={"id": "test-id-123", "section": "strategies", "content": "Test", "tags": []},
    )
    manager.apply_delta(delta_add)
    
    version_before = manager.playbook.version
    count_before = len(manager.playbook.bullets)
    
    # Replay the exact same ADD operation
    manager.apply_delta(delta_add)
    
    # Should be no-op: no version bump, no duplicate bullet
    assert manager.playbook.version == version_before
    assert len(manager.playbook.bullets) == count_before
    assert manager.playbook.bullets[0].id == "test-id-123"


def test_patch_same_content_idempotency():
    """PATCH with same content should still bump version (non-idempotent by design)."""
    manager = PlaybookManager()
    delta_add = DeltaOp(
        op="ADD", new_bullet={"section": "strategies", "content": "Original", "tags": []}
    )
    manager.apply_delta(delta_add)
    bullet_id = manager.playbook.bullets[0].id

    delta_patch = DeltaOp(op="PATCH", target_id=bullet_id, patch="Updated")
    manager.apply_delta(delta_patch)
    
    version_before = manager.playbook.version
    content_before = manager.playbook.bullets[0].content
    
    # Apply same patch again
    manager.apply_delta(delta_patch)
    
    # Content unchanged but version bumps (non-idempotent)
    assert manager.playbook.bullets[0].content == content_before
    assert manager.playbook.version == version_before + 1


def test_incr_helpful_non_idempotent():
    """INCR_HELPFUL is explicitly non-idempotent: each call increments."""
    manager = PlaybookManager()
    delta_add = DeltaOp(
        op="ADD", new_bullet={"section": "strategies", "content": "Test", "tags": []}
    )
    manager.apply_delta(delta_add)
    bullet_id = manager.playbook.bullets[0].id

    delta_incr = DeltaOp(op="INCR_HELPFUL", target_id=bullet_id)
    manager.apply_delta(delta_incr)
    manager.apply_delta(delta_incr)
    manager.apply_delta(delta_incr)

    # Each application increments (non-idempotent by design)
    assert manager.playbook.bullets[0].helpful == 3
    assert manager.playbook.version == 4


def test_incr_harmful_non_idempotent():
    """INCR_HARMFUL is explicitly non-idempotent: each call increments."""
    manager = PlaybookManager()
    delta_add = DeltaOp(
        op="ADD", new_bullet={"section": "strategies", "content": "Test", "tags": []}
    )
    manager.apply_delta(delta_add)
    bullet_id = manager.playbook.bullets[0].id

    delta_incr = DeltaOp(op="INCR_HARMFUL", target_id=bullet_id)
    manager.apply_delta(delta_incr)
    manager.apply_delta(delta_incr)

    # Each application increments (non-idempotent by design)
    assert manager.playbook.bullets[0].harmful == 2
    assert manager.playbook.version == 3


def test_incr_helpful_nonexistent_bullet_raises():
    manager = PlaybookManager()
    delta = DeltaOp(op="INCR_HELPFUL", target_id="nonexistent-id")

    with pytest.raises(ValueError, match="Bullet not found"):
        manager.apply_delta(delta)


def test_incr_harmful_nonexistent_bullet_raises():
    manager = PlaybookManager()
    delta = DeltaOp(op="INCR_HARMFUL", target_id="nonexistent-id")

    with pytest.raises(ValueError, match="Bullet not found"):
        manager.apply_delta(delta)


def test_deprecate_nonexistent_bullet_raises():
    manager = PlaybookManager()
    delta = DeltaOp(op="DEPRECATE", target_id="nonexistent-id")

    with pytest.raises(ValueError, match="Bullet not found"):
        manager.apply_delta(delta)


def test_add_duplicate_id_is_noop():
    """ADD with duplicate ID is a no-op (idempotent replay behavior)."""
    manager = PlaybookManager()
    delta1 = DeltaOp(
        op="ADD",
        new_bullet={"id": "test-id-1", "section": "strategies", "content": "First", "tags": []},
    )
    manager.apply_delta(delta1)

    version_before = manager.playbook.version
    count_before = len(manager.playbook.bullets)

    delta2 = DeltaOp(
        op="ADD",
        new_bullet={"id": "test-id-1", "section": "facts", "content": "Duplicate", "tags": []},
    )
    manager.apply_delta(delta2)

    # Should be no-op: version and count unchanged
    assert manager.playbook.version == version_before
    assert len(manager.playbook.bullets) == count_before
    assert manager.playbook.bullets[0].content == "First"  # Original content preserved
