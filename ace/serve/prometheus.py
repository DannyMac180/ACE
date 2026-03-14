"""Prometheus exporter helpers for the online ACE server."""

from collections.abc import Callable
from typing import Any

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily
from starlette.responses import Response

from ace.core.metrics import MetricsTracker


class ACEPrometheusCollector:
    """Collect current ACE server and playbook metrics at scrape time."""

    def __init__(
        self,
        server_getter: Callable[[], Any],
        metrics_tracker: MetricsTracker,
    ) -> None:
        self._server_getter = server_getter
        self._metrics_tracker = metrics_tracker

    def describe(self):
        """Declare metric descriptors without touching live server state."""
        yield CounterMetricFamily(
            "ace_online_requests_processed_total",
            "Total online adaptation requests processed by the current ACE server session.",
        )
        yield CounterMetricFamily(
            "ace_online_ops_applied_total",
            "Total delta operations applied by the current ACE server session.",
        )
        yield CounterMetricFamily(
            "ace_online_feedback_helpful_total",
            "Total helpful feedback operations observed by the current ACE server session.",
        )
        yield CounterMetricFamily(
            "ace_online_feedback_harmful_total",
            "Total harmful feedback operations observed by the current ACE server session.",
        )
        yield CounterMetricFamily(
            "ace_online_auto_refine_runs_total",
            "Total auto-refine passes executed by the current ACE server session.",
        )
        yield CounterMetricFamily(
            "ace_online_auto_refine_merged_total",
            "Total bullets merged by auto-refine during the current ACE server session.",
        )
        yield CounterMetricFamily(
            "ace_online_auto_refine_archived_total",
            "Total bullets archived by auto-refine during the current ACE server session.",
        )
        yield GaugeMetricFamily(
            "ace_online_avg_adaptation_milliseconds",
            "Average online adaptation latency for the current ACE server session.",
        )
        yield GaugeMetricFamily(
            "ace_online_warmup_bullets_loaded",
            "Number of bullets loaded into the ACE server warmup state.",
        )
        yield GaugeMetricFamily(
            "ace_playbook_version",
            "Current persisted ACE playbook version.",
        )
        yield GaugeMetricFamily(
            "ace_playbook_bullets",
            "Current number of bullets in the ACE playbook.",
        )
        yield GaugeMetricFamily(
            "ace_playbook_helpful_total",
            "Current total helpful counter across all ACE playbook bullets.",
        )
        yield GaugeMetricFamily(
            "ace_playbook_harmful_total",
            "Current total harmful counter across all ACE playbook bullets.",
        )
        yield CounterMetricFamily(
            "ace_validation_attempts_total",
            "Total ACE validation attempts grouped by schema type.",
            labels=["schema_type"],
        )
        yield CounterMetricFamily(
            "ace_validation_successes_total",
            "Total successful ACE validation attempts grouped by schema type.",
            labels=["schema_type"],
        )
        yield CounterMetricFamily(
            "ace_validation_failures_total",
            "Total failed ACE validation attempts grouped by schema type.",
            labels=["schema_type"],
        )
        yield CounterMetricFamily(
            "ace_validation_json_decode_errors_total",
            "Total JSON decode errors grouped by schema type.",
            labels=["schema_type"],
        )
        yield CounterMetricFamily(
            "ace_validation_schema_errors_total",
            "Total schema validation errors grouped by schema type.",
            labels=["schema_type"],
        )
        yield GaugeMetricFamily(
            "ace_validation_success_rate",
            "ACE validation success rate grouped by schema type.",
            labels=["schema_type"],
        )

    def collect(self):  # pragma: no cover - exercised via generate_latest in tests
        server = self._server_getter()
        stats = server.get_stats()
        playbook = server.store.load_playbook()

        requests_processed = CounterMetricFamily(
            "ace_online_requests_processed_total",
            "Total online adaptation requests processed by the current ACE server session.",
        )
        requests_processed.add_metric([], float(stats.requests_processed))
        yield requests_processed

        ops_applied = CounterMetricFamily(
            "ace_online_ops_applied_total",
            "Total delta operations applied by the current ACE server session.",
        )
        ops_applied.add_metric([], float(stats.total_ops_applied))
        yield ops_applied

        helpful_feedback = CounterMetricFamily(
            "ace_online_feedback_helpful_total",
            "Total helpful feedback operations observed by the current ACE server session.",
        )
        helpful_feedback.add_metric([], float(stats.helpful_feedback_count))
        yield helpful_feedback

        harmful_feedback = CounterMetricFamily(
            "ace_online_feedback_harmful_total",
            "Total harmful feedback operations observed by the current ACE server session.",
        )
        harmful_feedback.add_metric([], float(stats.harmful_feedback_count))
        yield harmful_feedback

        auto_refine_runs = CounterMetricFamily(
            "ace_online_auto_refine_runs_total",
            "Total auto-refine passes executed by the current ACE server session.",
        )
        auto_refine_runs.add_metric([], float(stats.auto_refine_runs))
        yield auto_refine_runs

        auto_refine_merged = CounterMetricFamily(
            "ace_online_auto_refine_merged_total",
            "Total bullets merged by auto-refine during the current ACE server session.",
        )
        auto_refine_merged.add_metric([], float(stats.auto_refine_merged))
        yield auto_refine_merged

        auto_refine_archived = CounterMetricFamily(
            "ace_online_auto_refine_archived_total",
            "Total bullets archived by auto-refine during the current ACE server session.",
        )
        auto_refine_archived.add_metric([], float(stats.auto_refine_archived))
        yield auto_refine_archived

        avg_adaptation = GaugeMetricFamily(
            "ace_online_avg_adaptation_milliseconds",
            "Average online adaptation latency for the current ACE server session.",
        )
        avg_adaptation.add_metric([], float(stats.avg_adaptation_ms))
        yield avg_adaptation

        warmup_bullets = GaugeMetricFamily(
            "ace_online_warmup_bullets_loaded",
            "Number of bullets loaded into the ACE server warmup state.",
        )
        warmup_bullets.add_metric([], float(stats.warmup_bullets_loaded))
        yield warmup_bullets

        playbook_version = GaugeMetricFamily(
            "ace_playbook_version",
            "Current persisted ACE playbook version.",
        )
        playbook_version.add_metric([], float(playbook.version))
        yield playbook_version

        playbook_bullets = GaugeMetricFamily(
            "ace_playbook_bullets",
            "Current number of bullets in the ACE playbook.",
        )
        playbook_bullets.add_metric([], float(len(playbook.bullets)))
        yield playbook_bullets

        helpful_total = GaugeMetricFamily(
            "ace_playbook_helpful_total",
            "Current total helpful counter across all ACE playbook bullets.",
        )
        helpful_total.add_metric([], float(sum(b.helpful for b in playbook.bullets)))
        yield helpful_total

        harmful_total = GaugeMetricFamily(
            "ace_playbook_harmful_total",
            "Current total harmful counter across all ACE playbook bullets.",
        )
        harmful_total.add_metric([], float(sum(b.harmful for b in playbook.bullets)))
        yield harmful_total

        validation_attempts = CounterMetricFamily(
            "ace_validation_attempts_total",
            "Total ACE validation attempts grouped by schema type.",
            labels=["schema_type"],
        )
        validation_successes = CounterMetricFamily(
            "ace_validation_successes_total",
            "Total successful ACE validation attempts grouped by schema type.",
            labels=["schema_type"],
        )
        validation_failures = CounterMetricFamily(
            "ace_validation_failures_total",
            "Total failed ACE validation attempts grouped by schema type.",
            labels=["schema_type"],
        )
        validation_json_errors = CounterMetricFamily(
            "ace_validation_json_decode_errors_total",
            "Total JSON decode errors grouped by schema type.",
            labels=["schema_type"],
        )
        validation_schema_errors = CounterMetricFamily(
            "ace_validation_schema_errors_total",
            "Total schema validation errors grouped by schema type.",
            labels=["schema_type"],
        )
        validation_success_rate = GaugeMetricFamily(
            "ace_validation_success_rate",
            "ACE validation success rate grouped by schema type.",
            labels=["schema_type"],
        )

        for schema_type in ("all", "reflection", "delta"):
            metrics = (
                self._metrics_tracker.get_metrics()
                if schema_type == "all"
                else self._metrics_tracker.get_metrics(schema_type=schema_type)
            )
            labels = [schema_type]
            validation_attempts.add_metric(labels, float(metrics.total_attempts))
            validation_successes.add_metric(labels, float(metrics.successful_parses))
            validation_failures.add_metric(labels, float(metrics.failed_parses))
            validation_json_errors.add_metric(labels, float(metrics.json_decode_errors))
            validation_schema_errors.add_metric(labels, float(metrics.schema_validation_errors))
            validation_success_rate.add_metric(labels, float(metrics.success_rate))

        yield validation_attempts
        yield validation_successes
        yield validation_failures
        yield validation_json_errors
        yield validation_schema_errors
        yield validation_success_rate


def build_metrics_registry(
    server_getter: Callable[[], Any],
    metrics_tracker: MetricsTracker,
) -> CollectorRegistry:
    """Build a per-app collector registry for Prometheus scrapes."""
    registry = CollectorRegistry(auto_describe=True)
    registry.register(ACEPrometheusCollector(server_getter, metrics_tracker))
    return registry


def metrics_response(registry: CollectorRegistry) -> Response:
    """Render Prometheus text format for the provided registry."""
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
