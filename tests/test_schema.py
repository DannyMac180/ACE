from ace.core.schema import Bullet, Playbook


def test_bullet_creation():
    bullet = Bullet(
        id="test-001",
        section="strategies",
        content="Test bullet",
        tags=["test"]
    )
    assert bullet.id == "test-001"
    assert bullet.section == "strategies"
    assert bullet.content == "Test bullet"
    assert bullet.tags == ["test"]

def test_playbook_creation():
    bullets = [Bullet(id="1", section="strategies", content="test")]
    playbook = Playbook(version=1, bullets=bullets)
    assert playbook.version == 1
    assert len(playbook.bullets) == 1
