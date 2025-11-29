"""Multi-epoch training runner for offline adaptation.

Implements the ACE paper's multi-epoch offline adaptation:
"ACE further supports multi-epoch adaptation, where the same queries are
revisited to progressively strengthen the context."

Includes regression gating as described in the paper: after each epoch,
evaluate on a held-out split and rollback if metrics regress.
"""

import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

from ace.core.config import TrainingConfig, load_config
from ace.core.merge import Delta as MergeDelta
from ace.core.merge import apply_delta
from ace.core.regression import RegressionDetector, RegressionThresholds
from ace.core.schema import Playbook
from ace.core.storage.store_adapter import Store
from ace.curator.curator import curate
from ace.eval.metrics import mean_reciprocal_rank, precision_at_k, recall_at_k
from ace.refine.runner import refine
from ace.reflector.reflector import Reflector
from ace.reflector.schema import Reflection

from .schema import TrainingResult, TrainingSample, TrainingState, TrainSample

logger = logging.getLogger(__name__)


class TrainingRunner:
    """Runs multi-epoch offline adaptation training.

    The training loop:
    1. Load samples from JSONL file
    2. For each epoch (1 to N):
       a. For each sample, run reflect→curate→commit
       b. Compute metrics if ground_truth is present
       c. Optionally run refine at end of epoch
       d. If regression gating enabled: evaluate on held-out split,
          rollback to pre-epoch playbook if regression detected
    3. Return final training state and metrics
    """

    def __init__(
        self,
        store: Store | None = None,
        reflector: Reflector | None = None,
        refine_after_epoch: bool = True,
        refine_threshold: float = 0.90,
        shuffle: bool = False,
        retrieval_k: int = 10,
        gate_on_regression: bool | None = None,
        max_regression_delta: float | None = None,
        held_out_path: str | None = None,
        regression_metrics: list[str] | None = None,
    ):
        """Initialize the training runner.

        Args:
            store: Playbook store (loads from config if None)
            reflector: Reflector instance (creates default if None)
            refine_after_epoch: Whether to run refine after each epoch
            refine_threshold: Threshold for refine deduplication
            shuffle: Whether to shuffle samples each epoch
            retrieval_k: k value for precision/recall metrics
            gate_on_regression: Enable regression gating (rollback on regression)
            max_regression_delta: Max allowed regression before rollback (0.0-1.0)
            held_out_path: Path to held-out evaluation split (JSONL)
            regression_metrics: Metrics to check for regression
        """
        config = load_config()
        if store is None:
            store = Store(config.database.url)
        self.store = store
        self.reflector = reflector or Reflector()
        self.refine_after_epoch = refine_after_epoch
        self.refine_threshold = refine_threshold
        self.shuffle = shuffle
        self.retrieval_k = retrieval_k

        train_cfg: TrainingConfig = config.training
        self.gate_on_regression = (
            gate_on_regression if gate_on_regression is not None
            else train_cfg.gate_on_regression
        )
        self.max_regression_delta = (
            max_regression_delta if max_regression_delta is not None
            else train_cfg.max_regression_delta
        )
        self.held_out_path = (
            held_out_path if held_out_path is not None
            else train_cfg.held_out_path
        )
        self.regression_metrics = (
            regression_metrics if regression_metrics is not None
            else train_cfg.regression_metrics
        )

        self._epoch_metrics: dict[int, dict] = {}
        self._held_out_samples: list[TrainSample] | None = None
        self._regression_detector: RegressionDetector | None = None
        self._playbook_snapshots: dict[int, Playbook] = {}

    def load_samples(self, data_path: str) -> list[TrainingSample]:
        """Load training samples from a JSONL file.

        Each line should be a JSON object with:
        - id: Unique sample identifier
        - query: The task query
        - retrieved_bullet_ids: List of bullet IDs used (optional)
        - code_diff: Code changes (optional)
        - test_output: Test output (optional)
        - logs: Execution logs (optional)
        - env_meta: Additional metadata (optional)
        - success: Whether the task succeeded (optional)

        Args:
            data_path: Path to JSONL file

        Returns:
            List of TrainingSample objects
        """
        samples = []
        path = Path(data_path)

        if not path.exists():
            raise FileNotFoundError(f"Training data file not found: {data_path}")

        with open(path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if "id" not in data:
                        data["id"] = f"sample-{line_num}"
                    samples.append(TrainingSample.model_validate(data))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
                except Exception as e:
                    logger.warning(f"Skipping invalid sample on line {line_num}: {e}")

        logger.info(f"Loaded {len(samples)} training samples from {data_path}")
        return samples

    def load_train_samples(self, data_path: str) -> list[TrainSample]:
        """Load TrainSample format from JSONL (supports labeled/unlabeled modes).

        Format per line:
        - query: str (required)
        - input: dict | null (optional structured input)
        - ground_truth: str | dict | null (if present, enables metric computation)
        - feedback: dict | null (code_diff, test_output, logs, env_meta, success)

        Args:
            data_path: Path to JSONL file

        Returns:
            List of TrainSample objects
        """
        samples = []
        path = Path(data_path)

        if not path.exists():
            raise FileNotFoundError(f"Training data file not found: {data_path}")

        with open(path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    samples.append(TrainSample.model_validate(data))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
                except Exception as e:
                    logger.warning(f"Skipping invalid sample on line {line_num}: {e}")

        labeled_count = sum(1 for s in samples if s.is_labeled)
        logger.info(
            f"Loaded {len(samples)} samples ({labeled_count} labeled, "
            f"{len(samples) - labeled_count} unlabeled) from {data_path}"
        )
        return samples

    def compute_retrieval_metrics(
        self,
        samples: list[TrainSample],
        retrieved_results: list[list[str]],
    ) -> dict:
        """Compute retrieval metrics for labeled samples.

        Args:
            samples: List of samples with ground_truth
            retrieved_results: Retrieved bullet IDs per sample

        Returns:
            Dictionary of metrics (MRR, Recall@k, Precision@k)
        """
        labeled_samples = [
            (s, r)
            for s, r in zip(samples, retrieved_results, strict=False)
            if s.is_labeled
        ]
        if not labeled_samples:
            return {}

        relevant_ids = []
        ranked_results = []

        for sample, retrieved in labeled_samples:
            gt = sample.ground_truth
            if isinstance(gt, dict):
                ids = set(gt.get("relevant_bullet_ids", []))
            elif isinstance(gt, str):
                ids = {gt}
            else:
                ids = set()
            relevant_ids.append(ids)
            ranked_results.append(retrieved)

        if not ranked_results:
            return {}

        return {
            "mrr": mean_reciprocal_rank(ranked_results, relevant_ids),
            f"recall@{self.retrieval_k}": recall_at_k(
                ranked_results, relevant_ids, self.retrieval_k
            ),
            f"precision@{self.retrieval_k}": precision_at_k(
                ranked_results, relevant_ids, self.retrieval_k
            ),
        }

    def get_epoch_metrics(self, epoch: int) -> dict:
        """Get computed metrics for an epoch."""
        return self._epoch_metrics.get(epoch, {})

    def _init_regression_gating(self) -> None:
        """Initialize regression gating components."""
        if not self.gate_on_regression:
            return

        if not self.held_out_path:
            logger.warning(
                "Regression gating enabled but no held_out_path specified. "
                "Disabling regression gating."
            )
            self.gate_on_regression = False
            return

        try:
            self._held_out_samples = self.load_train_samples(self.held_out_path)
            logger.info(
                f"Loaded {len(self._held_out_samples)} held-out samples for regression gating"
            )
        except FileNotFoundError:
            logger.warning(
                f"Held-out file not found: {self.held_out_path}. "
                "Disabling regression gating."
            )
            self.gate_on_regression = False
            return

        thresholds = RegressionThresholds(
            success_rate_drop=self.max_regression_delta,
            retrieval_precision_drop=self.max_regression_delta,
        )
        self._regression_detector = RegressionDetector(thresholds)

    def _snapshot_playbook(self, epoch: int) -> None:
        """Take a snapshot of the current playbook before an epoch."""
        playbook = self.store.load_playbook()
        self._playbook_snapshots[epoch] = Playbook(
            version=playbook.version,
            bullets=[b.model_copy() for b in playbook.bullets],
        )
        logger.debug(f"Snapshot playbook v{playbook.version} before epoch {epoch}")

    def _restore_playbook(self, epoch: int) -> bool:
        """Restore playbook to pre-epoch snapshot.

        Returns:
            True if restoration succeeded, False otherwise.
        """
        if epoch not in self._playbook_snapshots:
            logger.error(f"No snapshot for epoch {epoch}, cannot restore")
            return False

        snapshot = self._playbook_snapshots[epoch]
        self.store.load_playbook_data(snapshot)
        logger.info(f"Restored playbook to v{snapshot.version} (pre-epoch {epoch})")
        return True

    def _evaluate_held_out(self, epoch: int) -> dict:
        """Evaluate current playbook on held-out split.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of computed metrics
        """
        if not self._held_out_samples:
            return {}

        from ace.core.retrieve import Retriever
        retriever = Retriever(self.store)

        retrieved_results = []
        for sample in self._held_out_samples:
            bullets = retriever.retrieve(sample.query, top_k=self.retrieval_k)
            retrieved_results.append([b.id for b in bullets])

        metrics = self.compute_retrieval_metrics(
            self._held_out_samples, retrieved_results
        )
        self._epoch_metrics[epoch] = metrics
        logger.info(f"Epoch {epoch} held-out metrics: {metrics}")
        return metrics

    def _check_regression(self, epoch: int, metrics: dict) -> bool:
        """Check if current epoch metrics regressed.

        Args:
            epoch: Current epoch number
            metrics: Computed metrics for this epoch

        Returns:
            True if regression detected, False otherwise
        """
        if not self._regression_detector or not metrics:
            return False

        regression_detected = False
        for metric_name in self.regression_metrics:
            normalized_name = metric_name.lower().strip()
            matching_key = None
            for key in metrics:
                if normalized_name in key.lower():
                    matching_key = key
                    break

            if matching_key is None:
                continue

            value = metrics[matching_key]
            report = self._regression_detector.detect_regression(
                benchmark_name="held_out",
                metric_name=matching_key,
                current_value=value,
                higher_is_better=True,
            )

            if report.detected:
                logger.warning(
                    f"REGRESSION epoch {epoch}: {report.message}"
                )
                regression_detected = True
            else:
                self._regression_detector.record_result(
                    benchmark_name="held_out",
                    metric_name=matching_key,
                    value=value,
                    metadata={"epoch": epoch},
                )

        return regression_detected

    def train(
        self,
        data_path: str,
        epochs: int = 1,
        resume_state: TrainingState | None = None,
    ) -> TrainingResult:
        """Run multi-epoch training.

        Args:
            data_path: Path to JSONL file with training samples
            epochs: Number of epochs to run
            resume_state: Optional state to resume from

        Returns:
            TrainingResult with metrics and final state
        """
        start_time = time.time()
        samples = self.load_samples(data_path)

        if not samples:
            raise ValueError("No valid training samples found")

        self._init_regression_gating()

        state = resume_state or TrainingState(total_epochs=epochs)
        state.total_epochs = epochs

        playbook = self.store.load_playbook()
        version_start = playbook.version
        total_ops = 0
        total_samples = 0
        epochs_rolled_back = 0

        start_epoch_num = max(1, state.current_epoch)
        if state.is_epoch_completed(start_epoch_num):
            start_epoch_num += 1

        for epoch_num in range(start_epoch_num, epochs + 1):
            is_resuming = state.is_epoch_in_progress(epoch_num)

            if is_resuming:
                logger.info(f"Resuming epoch {epoch_num}/{epochs}")
            else:
                logger.info(f"Starting epoch {epoch_num}/{epochs}")
                playbook = self.store.load_playbook()
                state.start_epoch(epoch_num, playbook.version)

            if self.gate_on_regression and not is_resuming:
                self._snapshot_playbook(epoch_num)

            epoch_samples = list(samples)
            if self.shuffle and not is_resuming:
                import random

                random.shuffle(epoch_samples)

            processed = state.get_processed_samples_for_epoch(epoch_num)
            samples_to_process = [s for s in epoch_samples if s.id not in processed]

            epoch_ops = 0
            for sample in samples_to_process:
                ops_applied, new_version = self._process_sample(sample, epoch_num)
                state.record_sample(
                    sample_id=sample.id,
                    epoch=epoch_num,
                    ops_applied=ops_applied,
                    version_before=playbook.version,
                    version_after=new_version,
                )
                epoch_ops += ops_applied
                playbook = self.store.load_playbook()

            total_ops += epoch_ops
            total_samples += len(samples_to_process)

            if self.refine_after_epoch:
                self._run_refine()

            playbook = self.store.load_playbook()

            if self.gate_on_regression:
                metrics = self._evaluate_held_out(epoch_num)
                if self._check_regression(epoch_num, metrics):
                    logger.warning(
                        f"Rolling back epoch {epoch_num} due to regression "
                        f"(playbook v{playbook.version})"
                    )
                    if self._restore_playbook(epoch_num):
                        playbook = self.store.load_playbook()
                        epochs_rolled_back += 1
                        logger.info(
                            f"ROLLBACK: Epoch {epoch_num} reverted, "
                            f"playbook now v{playbook.version}"
                        )
                    else:
                        logger.error(
                            f"Failed to rollback epoch {epoch_num}, keeping changes"
                        )
                else:
                    logger.info(
                        f"KEEP: Epoch {epoch_num} passed regression gate, "
                        f"playbook v{playbook.version}"
                    )

            state.complete_epoch(
                epoch=epoch_num,
                samples_processed=len(samples_to_process),
                ops_applied=epoch_ops,
                playbook_version=playbook.version,
            )

            logger.info(
                f"Epoch {epoch_num} complete: {len(samples_to_process)} samples, "
                f"{epoch_ops} ops, version {playbook.version}"
            )

        state.completed_at = datetime.now(UTC)
        playbook = self.store.load_playbook()

        duration = time.time() - start_time
        return TrainingResult(
            epochs_completed=epochs,
            total_samples_processed=total_samples,
            total_ops_applied=total_ops,
            playbook_version_start=version_start,
            playbook_version_end=playbook.version,
            duration_seconds=duration,
            state=state,
        )

    def _process_sample(self, sample: TrainingSample, epoch: int) -> tuple[int, int]:
        """Process a single training sample through reflect→curate→commit.

        Args:
            sample: The training sample to process
            epoch: Current epoch number

        Returns:
            Tuple of (ops_applied, new_playbook_version)
        """
        logger.debug(f"Processing sample {sample.id} in epoch {epoch}")

        try:
            reflection = self.reflector.reflect(
                query=sample.query,
                retrieved_bullet_ids=sample.retrieved_bullet_ids,
                code_diff=sample.code_diff,
                test_output=sample.test_output,
                logs=sample.logs,
                env_meta=sample.env_meta,
            )

            playbook = self.store.load_playbook()
            delta = curate(reflection, existing_bullets=playbook.bullets)

            if not delta.ops:
                logger.debug(f"No ops generated for sample {sample.id}")
                return 0, playbook.version

            merge_delta = MergeDelta.from_dict(delta.model_dump())
            new_playbook = apply_delta(playbook, merge_delta, self.store)

            logger.debug(
                f"Applied {len(delta.ops)} ops for sample {sample.id}, "
                f"version {playbook.version} -> {new_playbook.version}"
            )
            return len(delta.ops), new_playbook.version

        except Exception as e:
            logger.error(f"Error processing sample {sample.id}: {e}")
            playbook = self.store.load_playbook()
            return 0, playbook.version

    def _run_refine(self) -> None:
        """Run refine to deduplicate and consolidate bullets."""
        logger.debug("Running post-epoch refine")
        playbook = self.store.load_playbook()
        empty_reflection = Reflection()
        result = refine(empty_reflection, playbook, threshold=self.refine_threshold)

        for bullet in playbook.bullets:
            self.store.save_bullet(bullet)

        logger.debug(f"Refine: merged={result.merged}, archived={result.archived}")

    def save_state(self, state: TrainingState, path: str) -> None:
        """Save training state to a JSON file for resumption.

        Args:
            state: Training state to save
            path: Path to save state file
        """
        with open(path, "w") as f:
            json.dump(state.model_dump(), f, indent=2, default=str)
        logger.info(f"Saved training state to {path}")

    def load_state(self, path: str) -> TrainingState:
        """Load training state from a JSON file.

        Args:
            path: Path to state file

        Returns:
            TrainingState loaded from file
        """
        with open(path) as f:
            data = json.load(f)
        return TrainingState.model_validate(data)
