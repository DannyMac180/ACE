#!/usr/bin/env python3
"""Core logic for the 'ace.stats' MCP tool."""

from ace.core.store import Store


def run_stats() -> dict:
    """Retrieve and return playbook statistics.

    Returns:
        Dict containing version, total_bullets, helpful_ratio, etc.
    """
    store = Store()

    bullets = store.get_bullets()
    version = store.get_version()
    total_bullets = len(bullets)

    total_helpful = sum(b.helpful for b in bullets)
    total_harmful = sum(b.harmful for b in bullets)
    helpful_ratio = total_helpful / max(1, total_helpful + total_harmful)

    stats = {
        "version": version,
        "num_bullets": total_bullets,
        "helpful_ratio": round(helpful_ratio, 3),
        "total_helpful": total_helpful,
        "total_harmful": total_harmful,
    }

    print(f"Playbook Version: {version}")
    print(f"Total Bullet Count: {total_bullets}")
    print(f"Helpful Ratio: {helpful_ratio:.3f}")
    print(f"Total Helpful: {total_helpful}")
    print(f"Total Harmful: {total_harmful}")

    return stats


if __name__ == "__main__":
    run_stats()
