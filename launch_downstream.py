"""Launch 1% and 10% downstream suites in parallel Modal apps.

Each budget runs as its own detached Modal app (le-satclr-1pct,
le-satclr-10pct) so both run concurrently. Each app runs
downstream_remote, which runs probe, finetune, baseline sequentially
from the same SSL checkpoint on one GPU.
"""
import argparse
import concurrent.futures
import os
import subprocess


def build_command(encoder_checkpoint, epochs, batch_size, label_percent,
                  max_batches, policy, ssl_scope, isolated_tracking):
    command = [
        "modal", "run", "--detach",
        "modal_app.py::downstream_remote",
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--encoder-checkpoint", str(encoder_checkpoint),
        "--max-batches", str(max_batches),
        "--policy", policy,
        "--ssl-scope", ssl_scope,
    ]
    if isolated_tracking:
        command.append("--isolated-tracking")
    # Keep budget last: tests and app-name derivation rely on command[-1].
    command += ["--label-percent", str(label_percent)]
    return command


def _run_budget(budget, **kwargs):
    command = build_command(label_percent=budget, **kwargs)
    env = dict(os.environ)
    env["LE_SATCLR_APP_NAME"] = f"le-satclr-{budget}pct"
    result = subprocess.run(command, env=env)
    return budget, result.returncode


def launch(encoder_checkpoint, epochs=0, batch_size=0, max_batches=0,
           policy="standard", ssl_scope="all", isolated_tracking=False):
    params = dict(encoder_checkpoint=str(encoder_checkpoint), epochs=epochs,
                  batch_size=batch_size, max_batches=max_batches,
                  policy=policy, ssl_scope=ssl_scope,
                  isolated_tracking=isolated_tracking)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(_run_budget, budget, **params): budget
                   for budget in (1, 10)}
        codes = {}
        for future in concurrent.futures.as_completed(futures):
            budget, code = future.result()
            codes[budget] = code
    failed = sorted(b for b, c in codes.items() if c != 0)
    if failed:
        details = ", ".join(f"{b}% (exit {codes[b]})" for b in failed)
        raise RuntimeError(f"downstream failed for budgets: {details}")
    return codes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssl-checkpoint", required=True)
    parser.add_argument("--epochs", type=int, default=0,
                        help="0 selects per-stage defaults: simclr 100, probe 50, finetune 75, baseline 100")
    parser.add_argument("--batch-size", type=int, default=0,
                        help="0 selects per-stage defaults: simclr 128, downstream 64")
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--policy", default="standard")
    parser.add_argument("--ssl-scope", default="all")
    parser.add_argument("--isolated-tracking", action="store_true")
    args = parser.parse_args()
    launch(args.ssl_checkpoint, epochs=args.epochs,
           batch_size=args.batch_size, max_batches=args.max_batches,
           policy=args.policy, ssl_scope=args.ssl_scope,
           isolated_tracking=args.isolated_tracking)
