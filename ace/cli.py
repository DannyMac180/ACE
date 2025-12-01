"""ACE CLI entrypoint."""

import argparse
import json
import sys
from dataclasses import asdict
from typing import Any, NoReturn

from ace.core.config import load_config
from ace.core.merge import Delta as MergeDelta
from ace.core.merge import apply_delta
from ace.core.retrieve import Retriever
from ace.core.schema import Playbook
from ace.core.storage.store_adapter import Store
from ace.curator.curator import curate
from ace.pipeline import Pipeline
from ace.refine.runner import refine
from ace.reflector.reflector import Reflector
from ace.reflector.schema import Reflection
from ace.serve.runner import run_server as run_online_server
from ace.train.runner import TrainingRunner


def read_json_input(path_or_stdin: str | None) -> dict[str, Any]:
    """Read JSON from file path or stdin."""
    if path_or_stdin and path_or_stdin != "-":
        with open(path_or_stdin) as f:
            return json.load(f)  # type: ignore
    else:
        return json.load(sys.stdin)  # type: ignore


def print_output(data: Any, as_json: bool) -> None:
    """Print output as JSON or human-readable format."""
    if as_json:
        json.dump(data, sys.stdout, indent=2, default=str)
        print()
    else:
        if isinstance(data, dict):
            for key, value in data.items():
                print(f"{key}: {value}")
        elif isinstance(data, list):
            for item in data:
                print(item)
        else:
            print(data)


def cmd_retrieve(args: argparse.Namespace) -> None:
    """Retrieve bullets matching a query."""
    config = load_config()
    store = Store(config.database.url)
    retriever = Retriever(store)
    bullets = retriever.retrieve(args.query, top_k=args.top_k)

    if args.json:
        print_output([b.model_dump() for b in bullets], as_json=True)
    else:
        print(f"Found {len(bullets)} bullets:")
        for bullet in bullets:
            print(f"\n[{bullet.id}] ({bullet.section})")
            print(f"  {bullet.content}")
            print(f"  Tags: {', '.join(bullet.tags)}")
            print(f"  Helpful: {bullet.helpful} | Harmful: {bullet.harmful}")


def cmd_reflect(args: argparse.Namespace) -> None:
    """Generate reflection from task execution data."""
    from ace.generator.schemas import TrajectoryDoc

    doc = read_json_input(args.doc)

    trajectory_doc = TrajectoryDoc(
        query=doc.get("query", ""),
        retrieved_bullet_ids=doc.get("retrieved_bullet_ids", []),
        code_diff=doc.get("code_diff", ""),
        test_output=doc.get("test_output", ""),
        logs=doc.get("logs", ""),
        env_meta=doc.get("env_meta") or {},
    )

    config = load_config()
    num_passes = args.passes if args.passes is not None else config.reflector.passes
    similarity_threshold = config.reflector.similarity_threshold

    reflector = Reflector()

    if num_passes > 1:
        reflection = reflector.reflect_multi(
            trajectory_doc,
            num_passes=num_passes,
            similarity_threshold=similarity_threshold,
        )
    else:
        reflection = reflector.reflect(trajectory_doc)

    print_output(asdict(reflection), as_json=args.json)


def cmd_curate(args: argparse.Namespace) -> None:
    """Convert reflection to delta operations."""
    reflection_data = read_json_input(args.reflection)
    reflection = Reflection(**reflection_data)

    delta = curate(reflection)
    print_output(delta.model_dump(), as_json=args.json)


def cmd_commit(args: argparse.Namespace) -> None:
    """Apply delta operations to playbook."""
    config = load_config()
    store = Store(config.database.url)
    delta_data = read_json_input(args.delta)
    delta = MergeDelta.from_dict(delta_data)

    if args.dry_run:
        print("DRY RUN - would apply the following operations:")
        for op in delta.ops:
            print(f"  {op.op_type}: {op.data}")
        playbook = store.load_playbook()
        print(f"\nCurrent version: {playbook.version}")
        print(f"Would increment to: {playbook.version + 1}")
    else:
        playbook = store.load_playbook()
        new_playbook = apply_delta(playbook, delta, store)
        result: dict[str, Any] = {"version": new_playbook.version}
        print_output(result, as_json=args.json)


def cmd_playbook_dump(args: argparse.Namespace) -> None:
    """Dump full playbook JSON."""
    config = load_config()
    store = Store(config.database.url)
    playbook = store.load_playbook()

    playbook_json = playbook.model_dump()

    if args.out:
        with open(args.out, "w") as f:
            json.dump(playbook_json, f, indent=2, default=str)
        print(f"Playbook exported to {args.out}")
    else:
        print_output(playbook_json, as_json=True)


def cmd_playbook_import(args: argparse.Namespace) -> None:
    """Import playbook JSON into the store."""
    config = load_config()
    store = Store(config.database.url)

    data = read_json_input(args.file)
    playbook = Playbook.model_validate(data)

    store.load_playbook_data(playbook)

    result: dict[str, Any] = {
        "version": playbook.version,
        "bullets_imported": len(playbook.bullets),
    }
    print_output(result, as_json=args.json)


def cmd_tag(args: argparse.Namespace) -> None:
    """Tag a bullet as helpful or harmful."""
    config = load_config()
    store = Store(config.database.url)

    op_type = "INCR_HELPFUL" if args.helpful else "INCR_HARMFUL"
    ops_list = []
    for _ in range(args.count):
        ops_list.append({"op": op_type, "target_id": args.bullet_id})

    delta = MergeDelta.from_dict({"ops": ops_list})
    playbook = store.load_playbook()
    new_playbook = apply_delta(playbook, delta, store)

    result: dict[str, Any] = {
        "version": new_playbook.version,
        "bullet_id": args.bullet_id,
        "tag": op_type,
    }
    print_output(result, as_json=args.json)


def cmd_evolve(args: argparse.Namespace) -> None:
    """Run full reflect→curate→commit pipeline."""
    from ace.generator.schemas import TrajectoryDoc

    doc = read_json_input(args.doc)

    trajectory_doc = TrajectoryDoc(
        query=doc.get("query", ""),
        retrieved_bullet_ids=doc.get("retrieved_bullet_ids", []),
        code_diff=doc.get("code_diff", ""),
        test_output=doc.get("test_output", ""),
        logs=doc.get("logs", ""),
        env_meta=doc.get("env_meta") or {},
    )

    reflector = Reflector()
    reflection = reflector.reflect(trajectory_doc)

    delta = curate(reflection)

    if args.print_delta:
        print("Generated Delta:")
        print_output(delta.model_dump(), as_json=True)

    if args.apply:
        config = load_config()
        store = Store(config.database.url)
        playbook = store.load_playbook()

        # Convert to MergeDelta for apply_delta
        merge_delta = MergeDelta.from_dict(delta.model_dump())
        new_playbook = apply_delta(playbook, merge_delta, store)
        result: dict[str, Any] = {
            "version": new_playbook.version,
            "ops_applied": len(delta.ops),
        }
        print_output(result, as_json=args.json)
    else:
        print("\nDRY RUN - use --apply to commit changes")
        print(f"Would apply {len(delta.ops)} operations")


def cmd_refine(args: argparse.Namespace) -> None:
    """Run refine to deduplicate and consolidate bullets."""
    config = load_config()
    store = Store(config.database.url)
    playbook = store.load_playbook()

    empty_reflection = Reflection()
    result = refine(empty_reflection, playbook, threshold=args.threshold)

    if not args.dry_run:
        for bullet in playbook.bullets:
            store.save_bullet(bullet)

    output = {"merged": result.merged, "archived": result.archived}
    if args.dry_run:
        output["dry_run"] = True

    print_output(output, as_json=args.json)


def cmd_stats(args: argparse.Namespace) -> None:
    """Show playbook statistics."""
    config = load_config()
    store = Store(config.database.url)
    playbook = store.load_playbook()

    total = len(playbook.bullets)
    helpful = sum(b.helpful for b in playbook.bullets)
    harmful = sum(b.harmful for b in playbook.bullets)
    helpful_ratio = helpful / max(helpful + harmful, 1)

    sections: dict[str, int] = {}
    for b in playbook.bullets:
        sections[b.section] = sections.get(b.section, 0) + 1

    result: dict[str, Any] = {
        "version": playbook.version,
        "total_bullets": total,
        "helpful": helpful,
        "harmful": harmful,
        "helpful_ratio": helpful_ratio,
        "sections": sections,
    }

    if args.json:
        print_output(result, as_json=True)
    else:
        print("Playbook Statistics:")
        print(f"  Version: {result['version']}")
        print(f"  Total bullets: {result['total_bullets']}")
        print(f"  Helpful: {result['helpful']}")
        print(f"  Harmful: {result['harmful']}")
        print(f"  Helpful ratio: {result['helpful_ratio']:.2%}")
        print("\nBullets by section:")
        for section, count in sorted(result['sections'].items()):
            print(f"  {section}: {count}")


def cmd_serve(args: argparse.Namespace) -> None:
    """Start the online serving server for test-time adaptation."""
    print(f"Starting ACE online server on {args.host}:{args.port}")
    print(f"  Mode: {'online' if args.online else 'offline'}")
    print(f"  Auto-adapt: {not args.no_adapt}")
    if args.warmup:
        print(f"  Warmup: {args.warmup}")
    if args.auto_refine_every > 0:
        print(f"  Auto-refine every: {args.auto_refine_every} deltas")
    if args.max_bullets is not None:
        print(f"  Max bullets: {args.max_bullets}")

    run_online_server(
        host=args.host,
        port=args.port,
        auto_adapt=not args.no_adapt,
        reload=args.reload,
        warmup_path=args.warmup,
        auto_refine_every=args.auto_refine_every,
        max_bullets=args.max_bullets,
    )


def cmd_train(args: argparse.Namespace) -> None:
    """Run multi-epoch offline adaptation training."""
    resume_state = None
    if args.resume:
        try:
            runner = TrainingRunner()
            resume_state = runner.load_state(args.resume)
            print(f"Resuming from epoch {resume_state.current_epoch + 1}")
        except FileNotFoundError:
            print(f"Warning: Resume state file not found: {args.resume}")
            print("Starting fresh training run")

    runner = TrainingRunner(
        refine_after_epoch=not args.no_refine,
        refine_threshold=args.refine_threshold,
        shuffle=args.shuffle,
    )

    print(f"Starting training: {args.epochs} epochs on {args.data}")

    result = runner.train(
        data_path=args.data,
        epochs=args.epochs,
        resume_state=resume_state,
    )

    if args.save_state:
        runner.save_state(result.state, args.save_state)

    output = {
        "epochs_completed": result.epochs_completed,
        "total_samples_processed": result.total_samples_processed,
        "total_ops_applied": result.total_ops_applied,
        "playbook_version_start": result.playbook_version_start,
        "playbook_version_end": result.playbook_version_end,
        "duration_seconds": round(result.duration_seconds, 2),
    }

    if args.json:
        print_output(output, as_json=True)
    else:
        print("\nTraining Complete:")
        print(f"  Epochs: {output['epochs_completed']}")
        print(f"  Samples processed: {output['total_samples_processed']}")
        print(f"  Ops applied: {output['total_ops_applied']}")
        v_start, v_end = output['playbook_version_start'], output['playbook_version_end']
        print(f"  Playbook version: {v_start} → {v_end}")
        print(f"  Duration: {output['duration_seconds']}s")


def cmd_pipeline(args: argparse.Namespace) -> None:
    """Run the full ACE pipeline: retrieve→generate→reflect→curate→merge."""
    config = load_config()
    pipeline = Pipeline(db_path=config.database.url)

    if args.feedback:
        doc = read_json_input(args.feedback)
        result = pipeline.run_with_feedback(
            query=args.query,
            code_diff=doc.get("code_diff", ""),
            test_output=doc.get("test_output", ""),
            logs=doc.get("logs", ""),
            env_meta=doc.get("env_meta"),
            auto_commit=not args.dry_run,
        )
    else:
        result = pipeline.run_full_cycle(
            query=args.query,
            auto_commit=not args.dry_run,
        )

    output: dict[str, Any] = {
        "playbook_version": result.playbook.version,
        "trajectory_steps": result.trajectory.total_steps,
        "trajectory_status": result.trajectory.final_status,
        "candidate_bullets": len(result.reflection.candidate_bullets),
        "bullet_tags": len(result.reflection.bullet_tags),
        "delta_ops_applied": result.delta_ops_applied,
        "retrieved_bullets": len(result.retrieved_bullets),
    }

    if args.dry_run:
        output["dry_run"] = True

    if args.json:
        print_output(output, as_json=True)
    else:
        print("Pipeline completed:")
        print(f"  Playbook version: {output['playbook_version']}")
        print(f"  Trajectory: {output['trajectory_steps']} steps ({output['trajectory_status']})")
        print(f"  Retrieved bullets: {output['retrieved_bullets']}")
        print(f"  Candidate bullets: {output['candidate_bullets']}")
        print(f"  Bullet tags: {output['bullet_tags']}")
        print(f"  Delta ops applied: {output['delta_ops_applied']}")
        if args.dry_run:
            print("  (DRY RUN - no changes committed)")


def cmd_eval_run(args: argparse.Namespace) -> None:
    """Run evaluation benchmarks."""
    try:
        from ace.eval.harness import EvalRunner
    except ImportError:
        print("Error: Evaluation harness not yet implemented.")
        print("Expected module: ace.eval.harness")
        sys.exit(1)

    runner = EvalRunner()

    # Run evaluation suite
    results = runner.run_suite(
        suite=args.suite,
        baseline_path=args.baseline,
        fail_on_regression=args.fail_on_regression,
    )

    # Format output
    if args.format == "json" or args.json:
        output_data = results
        if args.out:
            with open(args.out, "w") as f:
                json.dump(output_data, f, indent=2, default=str)
            print(f"Results written to {args.out}")
        else:
            print_output(output_data, as_json=True)
    elif args.format == "markdown":
        markdown_output = runner.format_markdown(results)
        if args.out:
            with open(args.out, "w") as f:
                f.write(markdown_output)
            print(f"Results written to {args.out}")
        else:
            print(markdown_output)
    else:  # text format
        runner.print_results(results)


def main() -> NoReturn:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        prog="ace",
        description="ACE (Agentic Context Engineering) - evolve LLM context via playbook deltas",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    retrieve_parser = subparsers.add_parser("retrieve", help="Retrieve bullets matching a query")
    retrieve_parser.add_argument("query", help="Query string")
    retrieve_parser.add_argument("--top-k", type=int, default=24, help="Number of results")
    retrieve_parser.add_argument("--json", action="store_true", help="Output as JSON")
    retrieve_parser.set_defaults(func=cmd_retrieve)

    reflect_parser = subparsers.add_parser("reflect", help="Generate reflection from task data")
    reflect_parser.add_argument("--doc", help="Path to JSON doc (or '-' for stdin)")
    reflect_parser.add_argument(
        "--passes",
        type=int,
        default=None,
        help="Number of reflection passes (default: from config)",
    )
    reflect_parser.add_argument("--json", action="store_true", help="Output as JSON")
    reflect_parser.set_defaults(func=cmd_reflect)

    curate_parser = subparsers.add_parser("curate", help="Convert reflection to delta")
    curate_parser.add_argument("--reflection", help="Path to reflection JSON (or '-' for stdin)")
    curate_parser.add_argument("--json", action="store_true", help="Output as JSON")
    curate_parser.set_defaults(func=cmd_curate)

    commit_parser = subparsers.add_parser("commit", help="Apply delta to playbook")
    commit_parser.add_argument("--delta", help="Path to delta JSON (or '-' for stdin)")
    commit_parser.add_argument("--dry-run", action="store_true", help="Preview without applying")
    commit_parser.add_argument("--json", action="store_true", help="Output as JSON")
    commit_parser.set_defaults(func=cmd_commit)

    playbook_parser = subparsers.add_parser("playbook", help="Playbook operations")
    playbook_subparsers = playbook_parser.add_subparsers(dest="playbook_cmd", required=True)
    playbook_dump = playbook_subparsers.add_parser("dump", help="Dump full playbook JSON")
    playbook_dump.add_argument("--out", help="Output file path (default: stdout)")
    playbook_dump.set_defaults(func=cmd_playbook_dump)

    playbook_import = playbook_subparsers.add_parser("import", help="Import playbook JSON")
    playbook_import.add_argument("--file", help="Input file path (or '-' for stdin)")
    playbook_import.add_argument("--json", action="store_true", help="Output as JSON")
    playbook_import.set_defaults(func=cmd_playbook_import)

    tag_parser = subparsers.add_parser("tag", help="Tag bullet as helpful or harmful")
    tag_parser.add_argument("bullet_id", help="Bullet ID to tag")
    tag_group = tag_parser.add_mutually_exclusive_group(required=True)
    tag_group.add_argument("--helpful", action="store_true", help="Mark as helpful")
    tag_group.add_argument("--harmful", action="store_true", help="Mark as harmful")
    tag_parser.add_argument("--count", type=int, default=1, help="Increment count")
    tag_parser.add_argument("--json", action="store_true", help="Output as JSON")
    tag_parser.set_defaults(func=cmd_tag)

    evolve_parser = subparsers.add_parser("evolve", help="Run reflect→curate→commit pipeline")
    evolve_parser.add_argument("--doc", help="Path to task JSON (or '-' for stdin)")
    evolve_parser.add_argument("--print-delta", action="store_true", help="Print delta")
    evolve_parser.add_argument("--apply", action="store_true", help="Apply changes (not dry-run)")
    evolve_parser.add_argument("--json", action="store_true", help="Output as JSON")
    evolve_parser.set_defaults(func=cmd_evolve)

    refine_parser = subparsers.add_parser("refine", help="Deduplicate and consolidate bullets")
    refine_parser.add_argument(
        "--threshold",
        type=float,
        default=0.90,
        help="Similarity threshold for dedup",
    )
    refine_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without applying changes",
    )
    refine_parser.add_argument("--json", action="store_true", help="Output as JSON")
    refine_parser.set_defaults(func=cmd_refine)

    stats_parser = subparsers.add_parser("stats", help="Show playbook statistics")
    stats_parser.add_argument("--json", action="store_true", help="Output as JSON")
    stats_parser.set_defaults(func=cmd_stats)

    pipeline_parser = subparsers.add_parser(
        "pipeline", help="Run full ACE cycle: retrieve→generate→reflect→curate→merge"
    )
    pipeline_parser.add_argument("query", help="Task or goal to execute")
    pipeline_parser.add_argument(
        "--feedback",
        help="Path to JSON with explicit feedback (code_diff, test_output, logs)",
    )
    pipeline_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without committing changes",
    )
    pipeline_parser.add_argument("--json", action="store_true", help="Output as JSON")
    pipeline_parser.set_defaults(func=cmd_pipeline)

    serve_parser = subparsers.add_parser(
        "serve", help="Start online server for test-time sequential adaptation"
    )
    serve_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    serve_parser.add_argument(
        "--online",
        action="store_true",
        default=True,
        help="Run in online mode (test-time adaptation with execution feedback only)",
    )
    serve_parser.add_argument(
        "--no-adapt",
        action="store_true",
        help="Disable automatic adaptation (retrieve only)",
    )
    serve_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable hot reload for development",
    )
    serve_parser.add_argument(
        "--warmup",
        metavar="PLAYBOOK_JSON",
        help="Path to playbook JSON file for warm-start (preload before accepting queries)",
    )
    serve_parser.add_argument(
        "--auto-refine-every",
        type=int,
        default=0,
        metavar="N",
        help="Run refine every N deltas (0 = disabled, default: 0)",
    )
    serve_parser.add_argument(
        "--max-bullets",
        type=int,
        default=None,
        metavar="N",
        help="Max bullets before triggering refine (default: from config)",
    )
    serve_parser.set_defaults(func=cmd_serve)

    train_parser = subparsers.add_parser(
        "train", help="Run multi-epoch offline adaptation training"
    )
    train_parser.add_argument(
        "--data",
        required=True,
        help="Path to JSONL file with training samples",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1)",
    )
    train_parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle samples at each epoch",
    )
    train_parser.add_argument(
        "--no-refine",
        action="store_true",
        help="Skip refine step after each epoch",
    )
    train_parser.add_argument(
        "--refine-threshold",
        type=float,
        default=0.90,
        help="Similarity threshold for refine deduplication (default: 0.90)",
    )
    train_parser.add_argument(
        "--resume",
        help="Path to training state file to resume from",
    )
    train_parser.add_argument(
        "--save-state",
        help="Path to save training state for resumption",
    )
    train_parser.add_argument("--json", action="store_true", help="Output as JSON")
    train_parser.set_defaults(func=cmd_train)

    eval_parser = subparsers.add_parser("eval", help="Run evaluation benchmarks")
    eval_subparsers = eval_parser.add_subparsers(dest="eval_cmd", required=True)

    eval_run = eval_subparsers.add_parser("run", help="Run evaluation suite")
    eval_run.add_argument(
        "--suite",
        choices=["retrieval", "reflection", "e2e", "all"],
        default="all",
        help="Benchmark suite to run",
    )
    eval_run.add_argument("--json", action="store_true", help="Output as JSON")
    eval_run.add_argument(
        "--format",
        choices=["text", "json", "markdown"],
        default="text",
        help="Output format",
    )
    eval_run.add_argument("--out", help="Output file path (default: stdout)")
    eval_run.add_argument("--baseline", help="Baseline JSON for regression detection")
    eval_run.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit with error code if regression detected",
    )
    eval_run.set_defaults(func=cmd_eval_run)

    args = parser.parse_args()
    args.func(args)
    sys.exit(0)


if __name__ == "__main__":
    main()
