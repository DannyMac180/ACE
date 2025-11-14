#!/usr/bin/env python3
"""Core logic for the 'ace.stats' MCP tool."""

from ace.core.metrics import get_tracker
from ace.core.storage.store_adapter import Store


def run_stats() -> dict:
    """Retrieve and return playbook statistics.

    Returns:
        Dict containing version, total_bullets, helpful_ratio, validation_metrics, etc.
    """
    store = Store()
    tracker = get_tracker()

    bullets = store.get_bullets()
    version = store.get_version()
    total_bullets = len(bullets)

    total_helpful = sum(b.helpful for b in bullets)
    total_harmful = sum(b.harmful for b in bullets)
    helpful_ratio = total_helpful / max(1, total_helpful + total_harmful)

    validation_metrics = tracker.get_metrics()

    stats = {
        "version": version,
        "num_bullets": total_bullets,
        "helpful_ratio": round(helpful_ratio, 3),
        "total_helpful": total_helpful,
        "total_harmful": total_harmful,
        "validation_metrics": validation_metrics.to_dict(),
    }

    print(f"Playbook Version: {version}")
    print(f"Total Bullet Count: {total_bullets}")
    print(f"Helpful Ratio: {helpful_ratio:.3f}")
    print(f"Total Helpful: {total_helpful}")
    print(f"Total Harmful: {total_harmful}")
    print("\nValidation Metrics:")
    print(f"  Total Attempts: {validation_metrics.total_attempts}")
    print(f"  Success Rate: {validation_metrics.success_rate:.1%}")
    print(f"  JSON Decode Errors: {validation_metrics.json_decode_errors}")
    print(f"  Schema Validation Errors: {validation_metrics.schema_validation_errors}")

    return stats


if __name__ == "__main__":
    run_stats()
