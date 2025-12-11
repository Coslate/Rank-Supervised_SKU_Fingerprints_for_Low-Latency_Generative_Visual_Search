#!/usr/bin/env python
import argparse
from typing import Dict, Iterable, List, Optional

import wandb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Relog W&B runs from one project into another project "
                    "so curves can be compared in a single project."
    )
    # Example path format: "entity/project/run_id"
    parser.add_argument(
        "--src_runs",
        type=str,
        nargs="+",
        required=True,
        help=(
            "List of source runs in the form 'entity/project/run_id'. "
            "You can copy this from the run URL."
        ),
    )
    parser.add_argument(
        "--dst_entity",
        type=str,
        required=True,
        help="Destination W&B entity (e.g., your username or team).",
    )
    parser.add_argument(
        "--dst_project",
        type=str,
        required=True,
        help="Destination W&B project name.",
    )
    parser.add_argument(
        "--run_name_prefix",
        type=str,
        default="relog_",
        help="Prefix for new run names in the destination project.",
    )
    parser.add_argument(
        "--keys",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Optional list of metric keys to copy. "
            "If not set, all non-internal keys will be copied."
        ),
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Optional limit on number of history rows to relog (for debugging).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="If set, do not actually log to W&B, just print what would happen.",
    )
    return parser.parse_args()


def iter_history(
    run: "wandb.apis.public.Run",
    keys: Optional[List[str]] = None,
) -> Iterable[Dict]:
    """
    Iterate over the history rows of a W&B run.

    Uses scan_history to avoid loading everything into memory at once.
    """
    if keys is not None:
        # Ensure reserved keys are included so we can keep the same steps.
        keys_with_internal = list(keys) + ["_step"]
        seen = set()
        keys_with_internal = [
            k for k in keys_with_internal if not (k in seen or seen.add(k))
        ]
        history_iter = run.scan_history(keys=keys_with_internal, page_size=1000)
    else:
        history_iter = run.scan_history(page_size=1000)
    return history_iter


def relog_single_run(
    src_path: str,
    dst_entity: str,
    dst_project: str,
    run_name_prefix: str = "relog_",
    keys: Optional[List[str]] = None,
    max_steps: Optional[int] = None,
    dry_run: bool = False,
) -> None:
    api = wandb.Api()
    src_run = api.run(src_path)
    print(f"=== Relogging source run: {src_path} ===")
    print(f"Original name: {src_run.name}, id: {src_run.id}")

    # Prepare destination run name
    dst_run_name = f"{run_name_prefix}{src_run.name or src_run.id}"

    # Copy config but drop internal keys
    config = dict(src_run.config)
    config.pop("_wandb", None)

    if dry_run:
        print(
            f"[DRY RUN] Would create run in {dst_entity}/{dst_project} "
            f"with name '{dst_run_name}'"
        )
        dst_run = None
    else:
        dst_run = wandb.init(
            entity=dst_entity,
            project=dst_project,
            name=dst_run_name,
            config=config,
            reinit=True,
        )

    # Iterate over history and log
    count = 0
    for row in iter_history(src_run, keys=keys):
        if max_steps is not None and count >= max_steps:
            break

        step = row.get("_step", count)

        # Drop internal keys starting with "_"
        log_data = {k: v for k, v in row.items() if not k.startswith("_")}

        # If user specified keys explicitly, filter to that subset.
        if keys is not None:
            log_data = {k: v for k, v in log_data.items() if k in keys}

        if not log_data:
            count += 1
            continue

        if dry_run:
            if count < 5:  # only print a few rows
                print(f"[DRY RUN] step={step}, data={log_data}")
        else:
            dst_run.log(log_data, step=int(step))

        count += 1

    if not dry_run and dst_run is not None:
        dst_run.finish()

    print(f"Relogged {count} history rows from {src_path} into {dst_project}.\n")


def main() -> None:
    args = parse_args()

    for src_path in args.src_runs:
        relog_single_run(
            src_path=src_path,
            dst_entity=args.dst_entity,
            dst_project=args.dst_project,
            run_name_prefix=args.run_name_prefix,
            keys=args.keys,
            max_steps=args.max_steps,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
