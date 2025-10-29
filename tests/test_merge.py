import pytest

from ace.core.manager import PlaybookManager
from ace.core.schema import Delta, DeltaBullet


def test_add_operation():
    manager = PlaybookManager()
    assert manager.playbook.version == 0
    assert len(manager.playbook.bullets) == 0

    delta = Delta(
        op="ADD",
        new_bullet=DeltaBullet(
            section="strategies", content="Use hybrid retrieval", tags=["topic:retrieval"]
        ),
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
    delta = Delta(op="ADD")

    with pytest.raises(ValueError, match="'ADD' operation requires new_bullet"):
        manager.apply_delta(delta)


def test_patch_operation():
    manager = PlaybookManager()

    delta_add = Delta(
        op="ADD", new_bullet=DeltaBullet(section="strategies", content="Original content", tags=[])
    )
    manager.apply_delta(delta_add)
    bullet_id = manager.playbook.bullets[0].id

    delta_patch = Delta(op="PATCH", target_id=bullet_id, patch="Updated content")
    manager.apply_delta(delta_patch)

    assert manager.playbook.version == 2
    assert manager.playbook.bullets[0].content == "Updated content"


def test_patch_without_target_raises():
    manager = PlaybookManager()
    delta = Delta(op="PATCH", patch="New content")

    with pytest.raises(ValueError, match="'PATCH' operation requires target_id and patch"):
        manager.apply_delta(delta)


def test_patch_nonexistent_bullet_raises():
    manager = PlaybookManager()
    delta = Delta(op="PATCH", target_id="nonexistent", patch="Content")

    with pytest.raises(ValueError, match="Bullet not found"):
        manager.apply_delta(delta)


def test_incr_helpful_operation():
    manager = PlaybookManager()

    delta_add = Delta(
        op="ADD", new_bullet=DeltaBullet(section="strategies", content="Test", tags=[])
    )
    manager.apply_delta(delta_add)
    bullet_id = manager.playbook.bullets[0].id

    delta_incr = Delta(op="INCR_HELPFUL", target_id=bullet_id)
    manager.apply_delta(delta_incr)

    assert manager.playbook.version == 2
    assert manager.playbook.bullets[0].helpful == 1
    assert manager.playbook.bullets[0].last_used is not None


def test_incr_harmful_operation():
    manager = PlaybookManager()

    delta_add = Delta(
        op="ADD", new_bullet=DeltaBullet(section="strategies", content="Test", tags=[])
    )
    manager.apply_delta(delta_add)
    bullet_id = manager.playbook.bullets[0].id

    delta_incr = Delta(op="INCR_HARMFUL", target_id=bullet_id)
    manager.apply_delta(delta_incr)

    assert manager.playbook.version == 2
    assert manager.playbook.bullets[0].harmful == 1


def test_deprecate_operation():
    manager = PlaybookManager()

    delta_add = Delta(
        op="ADD", new_bullet=DeltaBullet(section="strategies", content="Test", tags=[])
    )
    manager.apply_delta(delta_add)
    bullet_id = manager.playbook.bullets[0].id

    assert len(manager.playbook.bullets) == 1

    delta_deprecate = Delta(op="DEPRECATE", target_id=bullet_id)
    manager.apply_delta(delta_deprecate)

    assert manager.playbook.version == 2
    assert len(manager.playbook.bullets) == 0


def test_invalid_operation_raises():
    manager = PlaybookManager()
    delta = Delta(op="INVALID_OP")

    with pytest.raises(ValueError, match="Invalid operation: INVALID_OP"):
        manager.apply_delta(delta)


def test_multiple_operations_idempotent():
    manager = PlaybookManager()

    delta1 = Delta(
        op="ADD", new_bullet=DeltaBullet(section="strategies", content="Bullet 1", tags=[])
    )
    delta2 = Delta(
        op="ADD", new_bullet=DeltaBullet(section="facts", content="Bullet 2", tags=["tag1"])
    )

    manager.apply_delta(delta1)
    manager.apply_delta(delta2)

    assert manager.playbook.version == 2
    assert len(manager.playbook.bullets) == 2

    bullet_id = manager.playbook.bullets[0].id
    delta3 = Delta(op="INCR_HELPFUL", target_id=bullet_id)
    manager.apply_delta(delta3)

    assert manager.playbook.version == 3
    assert manager.playbook.bullets[0].helpful == 1
