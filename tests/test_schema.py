import pytest

from ace.core.schema import (
    SECTION_MIGRATION_MAP,
    Bullet,
    DeltaBullet,
    Playbook,
    normalize_section,
)


def test_bullet_creation():
    bullet = Bullet(
        id="test-001",
        section="strategies_and_hard_rules",
        content="Test bullet",
        tags=["test"],
    )
    assert bullet.id == "test-001"
    assert bullet.section == "strategies_and_hard_rules"
    assert bullet.content == "Test bullet"
    assert bullet.tags == ["test"]


def test_playbook_creation():
    bullets = [Bullet(id="1", section="strategies_and_hard_rules", content="test")]
    playbook = Playbook(version=1, bullets=bullets)
    assert playbook.version == 1
    assert len(playbook.bullets) == 1


# --- Backward Compatibility Tests ---


class TestSectionBackwardCompatibility:
    """Test that old section names are automatically migrated to new names."""

    def test_normalize_section_old_strategies(self):
        """Old 'strategies' should normalize to 'strategies_and_hard_rules'."""
        assert normalize_section("strategies") == "strategies_and_hard_rules"

    def test_normalize_section_old_templates(self):
        """Old 'templates' should normalize to 'code_snippets_and_templates'."""
        assert normalize_section("templates") == "code_snippets_and_templates"

    def test_normalize_section_old_code_snippets(self):
        """Old 'code_snippets' should normalize to 'code_snippets_and_templates'."""
        assert normalize_section("code_snippets") == "code_snippets_and_templates"

    def test_normalize_section_old_troubleshooting(self):
        """Old 'troubleshooting' should normalize to 'troubleshooting_and_pitfalls'."""
        assert normalize_section("troubleshooting") == "troubleshooting_and_pitfalls"

    def test_normalize_section_old_facts(self):
        """Old 'facts' should normalize to 'domain_facts_and_references'."""
        assert normalize_section("facts") == "domain_facts_and_references"

    def test_normalize_section_new_names_unchanged(self):
        """New section names should remain unchanged."""
        for new_name in [
            "strategies_and_hard_rules",
            "code_snippets_and_templates",
            "troubleshooting_and_pitfalls",
            "domain_facts_and_references",
        ]:
            assert normalize_section(new_name) == new_name

    def test_normalize_section_invalid_raises(self):
        """Invalid section names should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid section"):
            normalize_section("invalid_section")

    def test_bullet_with_old_section_strategies(self):
        """Bullet created with old 'strategies' should have normalized section."""
        bullet = Bullet(id="test-1", section="strategies", content="test")
        assert bullet.section == "strategies_and_hard_rules"

    def test_bullet_with_old_section_templates(self):
        """Bullet created with old 'templates' should have normalized section."""
        bullet = Bullet(id="test-2", section="templates", content="test")
        assert bullet.section == "code_snippets_and_templates"

    def test_bullet_with_old_section_code_snippets(self):
        """Bullet created with old 'code_snippets' should have normalized section."""
        bullet = Bullet(id="test-3", section="code_snippets", content="test")
        assert bullet.section == "code_snippets_and_templates"

    def test_bullet_with_old_section_troubleshooting(self):
        """Bullet created with old 'troubleshooting' should have normalized section."""
        bullet = Bullet(id="test-4", section="troubleshooting", content="test")
        assert bullet.section == "troubleshooting_and_pitfalls"

    def test_bullet_with_old_section_facts(self):
        """Bullet created with old 'facts' should have normalized section."""
        bullet = Bullet(id="test-5", section="facts", content="test")
        assert bullet.section == "domain_facts_and_references"

    def test_delta_bullet_with_old_section(self):
        """DeltaBullet created with old section name should have normalized section."""
        delta = DeltaBullet(section="strategies", content="test delta")
        assert delta.section == "strategies_and_hard_rules"

    def test_playbook_with_old_section_bullets(self):
        """Playbook loading bullets with old section names should normalize them."""
        bullets = [
            Bullet(id="1", section="strategies", content="strategy bullet"),
            Bullet(id="2", section="troubleshooting", content="troubleshooting bullet"),
            Bullet(id="3", section="facts", content="facts bullet"),
        ]
        playbook = Playbook(version=1, bullets=bullets)
        assert playbook.bullets[0].section == "strategies_and_hard_rules"
        assert playbook.bullets[1].section == "troubleshooting_and_pitfalls"
        assert playbook.bullets[2].section == "domain_facts_and_references"

    def test_migration_map_completeness(self):
        """Ensure migration map covers all old and new section names."""
        old_names = {"strategies", "templates", "code_snippets", "troubleshooting", "facts"}
        new_names = {
            "strategies_and_hard_rules",
            "code_snippets_and_templates",
            "troubleshooting_and_pitfalls",
            "domain_facts_and_references",
        }
        all_expected = old_names | new_names
        assert set(SECTION_MIGRATION_MAP.keys()) == all_expected
