#!/usr/bin/env python3
"""Control a synced Runpod checkout over SSH for reviewer-ablation runs.

The controller intentionally uses only SSH and rsync. It assumes the user has
already created or funded a Runpod pod and has SSH host/port/key details.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REMOTE_DIR = "/workspace/iterated_prefill_crawler"
LATEST_MARKER = "artifacts/out/runpod_latest_reviewer_ablation.txt"

DEFAULT_EXCLUDES = [
    ".git/",
    ".env",
    ".venv/",
    ".trio/",
    "__pycache__/",
    ".pytest_cache/",
    ".DS_Store",
    "artifacts/",
    "hf_models/",
    "outputs/",
    "multirun/",
    "*.pyc",
    "*.json.tmp",
]


class ConfigError(SystemExit):
    pass


def env_default(name: str, fallback: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else fallback


def require(value: str | None, message: str) -> str:
    if value:
        return value
    raise ConfigError(message)


def q(value: str | os.PathLike[str]) -> str:
    return shlex.quote(str(value))


def ssh_transport(args: argparse.Namespace) -> list[str]:
    transport = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "IdentitiesOnly=yes",
    ]
    if args.port:
        transport.extend(["-p", str(args.port)])
    if args.identity_file:
        transport.extend(["-i", str(Path(args.identity_file).expanduser())])
    return transport


def scp_transport(args: argparse.Namespace) -> list[str]:
    transport = [
        "scp",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "IdentitiesOnly=yes",
    ]
    if args.port:
        transport.extend(["-P", str(args.port)])
    if args.identity_file:
        transport.extend(["-i", str(Path(args.identity_file).expanduser())])
    return transport


def remote_target(args: argparse.Namespace) -> str:
    host = require(
        args.host,
        "Missing Runpod host. Pass --host or set RUNPOD_SSH_HOST.",
    )
    if host == "ssh.runpod.io":
        raise ConfigError(
            "RUNPOD_SSH_HOST=ssh.runpod.io is Runpod's interactive SSH proxy. "
            "The controller needs direct/full SSH with TCP forwarding, usually "
            "`ssh root@<public-ip> -p <mapped-port> -i <key>` from the Runpod "
            "Connect panel. Use the public IP as RUNPOD_SSH_HOST and the "
            "mapped SSH port as RUNPOD_SSH_PORT."
        )
    return f"{args.user}@{host}" if args.user else host


def remote_dir(args: argparse.Namespace) -> str:
    return require(
        args.remote_dir,
        "Missing remote directory. Pass --remote-dir or set RUNPOD_REMOTE_DIR.",
    )


def ssh_command(args: argparse.Namespace, remote_command: str) -> list[str]:
    return ssh_transport(args) + [remote_target(args), remote_command]


def run(
    cmd: Sequence[str],
    *,
    dry_run: bool = False,
    capture: bool = False,
    env: dict[str, str] | None = None,
) -> str:
    printable = " ".join(q(part) for part in cmd)
    print(f"+ {printable}")
    if dry_run:
        return ""
    if capture:
        proc = subprocess.run(
            list(cmd),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        if proc.stderr:
            print(proc.stderr, file=sys.stderr, end="")
        return proc.stdout
    subprocess.run(list(cmd), check=True, env=env)
    return ""


def run_ssh(args: argparse.Namespace, command: str, *, capture: bool = False) -> str:
    return run(ssh_command(args, command), dry_run=args.dry_run, capture=capture)


def build_rsync(args: argparse.Namespace, source: str, dest: str) -> list[str]:
    ssh = " ".join(q(part) for part in ssh_transport(args))
    cmd = ["rsync", "-az", "--human-readable", "-e", ssh]
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd + [source, dest]


def command_sync(args: argparse.Namespace) -> None:
    excludes = list(DEFAULT_EXCLUDES)
    if args.include_env:
        excludes.remove(".env")

    mkdir_cmd = f"mkdir -p {q(remote_dir(args))}"
    run_ssh(args, mkdir_cmd)

    if args.delete and args.transfer_method == "archive":
        raise ConfigError("--delete is only supported with --transfer-method rsync")

    rsync_cmd = build_rsync(
        args,
        f"{REPO_ROOT}/",
        f"{remote_target(args)}:{q(remote_dir(args).rstrip('/') + '/')}",
    )
    for pattern in excludes:
        rsync_cmd.insert(-2, f"--exclude={pattern}")
    if args.delete:
        rsync_cmd.insert(-2, "--delete")

    if args.transfer_method in ("auto", "rsync"):
        try:
            run(rsync_cmd, dry_run=args.dry_run)
            return
        except subprocess.CalledProcessError:
            if args.transfer_method == "rsync":
                raise
            if args.delete:
                raise ConfigError(
                    "rsync failed, and --delete cannot be honored by archive fallback"
                )
            print("rsync failed; falling back to archive+scp transfer", file=sys.stderr)

    sync_with_archive(args, excludes)


def tar_exclude_args(excludes: list[str]) -> list[str]:
    tar_args = []
    tar_args.append("--exclude=._*")
    for pattern in excludes:
        tar_args.append(f"--exclude={pattern.rstrip('/')}")
    return tar_args


def local_tar_create_prefix() -> list[str]:
    if sys.platform == "darwin":
        return ["tar", "--disable-copyfile", "--no-xattrs"]
    return ["tar"]


def sync_with_archive(args: argparse.Namespace, excludes: list[str]) -> None:
    remote_archive = f"/tmp/iterated_prefill_crawler_sync_{os.getpid()}.tar.gz"
    if args.dry_run:
        tar_cmd = (
            local_tar_create_prefix()
            + ["-czf", "/tmp/iterated_prefill_crawler_sync.tar.gz"]
            + tar_exclude_args(excludes)
            + ["-C", str(REPO_ROOT), "."]
        )
        scp_cmd = scp_transport(args) + [
            "/tmp/iterated_prefill_crawler_sync.tar.gz",
            f"{remote_target(args)}:{remote_archive}",
        ]
        extract_cmd = (
            f"mkdir -p {q(remote_dir(args))} "
            f"&& tar --no-same-owner -xzf {q(remote_archive)} -C {q(remote_dir(args))} "
            f"&& rm -f {q(remote_archive)}"
        )
        print("+ " + " ".join(q(part) for part in tar_cmd))
        print("+ " + " ".join(q(part) for part in scp_cmd))
        print("+ " + " ".join(q(part) for part in ssh_command(args, extract_cmd)))
        return

    archive_path = Path(tempfile.gettempdir()) / (
        f"iterated_prefill_crawler_sync_{os.getpid()}.tar.gz"
    )
    try:
        tar_cmd = (
            local_tar_create_prefix()
            + ["-czf", str(archive_path)]
            + tar_exclude_args(excludes)
            + ["-C", str(REPO_ROOT), "."]
        )
        tar_env = os.environ.copy()
        tar_env["COPYFILE_DISABLE"] = "1"
        run(tar_cmd, env=tar_env)
        scp_cmd = scp_transport(args) + [
            str(archive_path),
            f"{remote_target(args)}:{remote_archive}",
        ]
        run(scp_cmd)
        extract_cmd = (
            f"mkdir -p {q(remote_dir(args))} "
            f"&& tar --no-same-owner -xzf {q(remote_archive)} -C {q(remote_dir(args))} "
            f"&& rm -f {q(remote_archive)}"
        )
        run_ssh(args, extract_cmd)
    finally:
        archive_path.unlink(missing_ok=True)


def command_bootstrap(args: argparse.Namespace) -> None:
    cmd = (
        f"cd {q(remote_dir(args))} "
        "&& bash scripts/runpod_bootstrap.sh"
    )
    run_ssh(args, cmd)


def env_assignment(name: str, value: str | None) -> str | None:
    if value is None or value == "":
        return None
    return f"{name}={q(value)}"


def parse_env_pair(pair: str) -> tuple[str, str]:
    if "=" not in pair:
        raise ConfigError(f"--env must be NAME=VALUE, got: {pair}")
    name, value = pair.split("=", 1)
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ConfigError(f"Invalid environment variable name for --env: {name}")
    return name, value


def command_start(args: argparse.Namespace) -> None:
    assignments = [
        env_assignment("MODEL_CONFIG", args.model),
        env_assignment("CRAWLER_CONFIG", args.crawler),
        env_assignment("OUT_DIR", args.out_dir),
        env_assignment("SAMPLES", str(args.samples) if args.samples is not None else None),
        env_assignment(
            "EXTRA_OVERRIDES",
            " ".join(args.override) if args.override else None,
        ),
        env_assignment(
            "VALIDATE_ALL_DISCOVERED",
            "1" if args.validate_all_discovered else "0",
        ),
    ]
    for pair in args.env:
        name, value = parse_env_pair(pair)
        assignments.append(env_assignment(name, value))
    env_prefix = " ".join(part for part in assignments if part)
    inner = (
        f"cd {q(remote_dir(args))} "
        f"&& {env_prefix} bash scripts/runpod_reviewer_ready.sh"
    )
    remote = (
        f"cd {q(remote_dir(args))} && "
        f"if tmux has-session -t {q(args.session)} 2>/dev/null; then "
        f"echo 'tmux session already exists: {args.session}'; exit 2; "
        "fi && "
        "mkdir -p artifacts/log && "
        f"tmux new-session -d -s {q(args.session)} {q(inner)} && "
        f"echo 'started tmux session: {args.session}'"
    )
    run_ssh(args, remote)


def command_stop(args: argparse.Namespace) -> None:
    remote = (
        f"cd {q(remote_dir(args))} && "
        f"if tmux has-session -t {q(args.session)} 2>/dev/null; then "
        f"tmux kill-session -t {q(args.session)} && "
        f"echo 'stopped tmux session: {args.session}'; "
        "else "
        f"echo 'tmux session not found: {args.session}'; "
        "fi"
    )
    run_ssh(args, remote)


def command_status(args: argparse.Namespace) -> None:
    remote = (
        f"cd {q(remote_dir(args))} && "
        "echo 'tmux sessions:' && "
        f"(tmux ls 2>/dev/null | grep -F -- {q(args.session)} || true) && "
        "echo && echo 'latest output marker:' && "
        f"(cat {q(LATEST_MARKER)} 2>/dev/null || true) && "
        "latest=$(cat "
        f"{q(LATEST_MARKER)} 2>/dev/null || true); "
        "if [ -n \"$latest\" ] && [ -f \"$latest/run.log\" ]; then "
        "echo && echo 'recent run log:'; tail -n 40 \"$latest/run.log\"; "
        "fi"
    )
    run_ssh(args, remote)


def command_tail(args: argparse.Namespace) -> None:
    remote = (
        f"cd {q(remote_dir(args))} && "
        "latest=$(cat "
        f"{q(LATEST_MARKER)} 2>/dev/null || true); "
        "if [ -z \"$latest\" ] || [ ! -f \"$latest/run.log\" ]; then "
        "echo 'No latest run.log marker yet. Try status or attach to tmux.'; "
        "exit 1; "
        "fi; "
        f"tail -n {int(args.lines)} -f \"$latest/run.log\""
    )
    run_ssh(args, remote)


def get_remote_latest(args: argparse.Namespace) -> str:
    marker_cmd = f"cd {q(remote_dir(args))} && cat {q(LATEST_MARKER)}"
    latest = run_ssh(args, marker_cmd, capture=True).strip()
    if not latest:
        raise ConfigError(
            f"No latest marker found at {LATEST_MARKER}. "
            "Pass fetch --remote-out-dir explicitly if the run used a custom path."
        )
    return latest


def command_fetch(args: argparse.Namespace) -> None:
    remote_out = args.remote_out_dir or get_remote_latest(args)
    remote_name = Path(remote_out.rstrip("/")).name
    local_dir = Path(args.local_dir or REPO_ROOT / "artifacts" / "runpod" / remote_name)

    if remote_out.startswith("/"):
        remote_path = remote_out.rstrip("/") + "/"
    else:
        remote_path = f"{remote_dir(args).rstrip('/')}/{remote_out.rstrip('/')}/"

    rsync_cmd = build_rsync(
        args,
        f"{remote_target(args)}:{q(remote_path)}",
        f"{local_dir}/",
    )

    if args.transfer_method in ("auto", "rsync"):
        try:
            if not args.dry_run:
                local_dir.mkdir(parents=True, exist_ok=True)
            run(rsync_cmd, dry_run=args.dry_run)
            print(f"{'Would fetch' if args.dry_run else 'Fetched'} to: {local_dir}")
            return
        except subprocess.CalledProcessError:
            if args.transfer_method == "rsync":
                raise
            print("rsync failed; falling back to archive+scp transfer", file=sys.stderr)

    fetch_with_archive(args, remote_path, local_dir)


def fetch_with_archive(
    args: argparse.Namespace,
    remote_path: str,
    local_dir: Path,
) -> None:
    remote_archive = f"/tmp/iterated_prefill_crawler_fetch_{os.getpid()}.tar.gz"
    if args.dry_run:
        pack_cmd = (
            f"cd {q(remote_path)} "
            f"&& tar -czf {q(remote_archive)} ."
        )
        scp_cmd = scp_transport(args) + [
            f"{remote_target(args)}:{remote_archive}",
            "/tmp/iterated_prefill_crawler_fetch.tar.gz",
        ]
        unpack_cmd = [
            "tar",
            "-xzf",
            "/tmp/iterated_prefill_crawler_fetch.tar.gz",
            "-C",
            str(local_dir),
        ]
        print("+ " + " ".join(q(part) for part in ssh_command(args, pack_cmd)))
        print("+ " + " ".join(q(part) for part in scp_cmd))
        print("+ " + " ".join(q(part) for part in unpack_cmd))
        print(f"Would fetch to: {local_dir}")
        return

    local_dir.mkdir(parents=True, exist_ok=True)
    local_archive = Path(tempfile.gettempdir()) / (
        f"iterated_prefill_crawler_fetch_{os.getpid()}.tar.gz"
    )
    try:
        pack_cmd = (
            f"cd {q(remote_path)} "
            f"&& tar -czf {q(remote_archive)} ."
        )
        run_ssh(args, pack_cmd)
        scp_cmd = scp_transport(args) + [
            f"{remote_target(args)}:{remote_archive}",
            str(local_archive),
        ]
        run(scp_cmd)
        run(["tar", "-xzf", str(local_archive), "-C", str(local_dir)])
        run_ssh(args, f"rm -f {q(remote_archive)}")
        print(f"Fetched to: {local_dir}")
    finally:
        local_archive.unlink(missing_ok=True)


def add_connection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default=env_default("RUNPOD_SSH_HOST"))
    parser.add_argument("--port", default=env_default("RUNPOD_SSH_PORT"))
    parser.add_argument("--user", default=env_default("RUNPOD_SSH_USER", "root"))
    parser.add_argument("--identity-file", default=env_default("RUNPOD_SSH_KEY"))
    parser.add_argument(
        "--remote-dir",
        default=env_default("RUNPOD_REMOTE_DIR", DEFAULT_REMOTE_DIR),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print SSH/rsync commands without executing them.",
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Control a Runpod checkout over SSH for the reviewer-ready 2x2 run."
        )
    )
    add_connection_args(parser)

    sub = parser.add_subparsers(dest="command", required=True)

    sync = sub.add_parser("sync", help="rsync this checkout to the Runpod")
    sync.add_argument(
        "--include-env",
        action="store_true",
        help="Also sync .env. By default secrets are excluded.",
    )
    sync.add_argument(
        "--delete",
        action="store_true",
        help="Delete remote files absent locally, after applying excludes.",
    )
    sync.add_argument(
        "--transfer-method",
        choices=("auto", "rsync", "archive"),
        default="auto",
        help="Transfer implementation. auto tries rsync then archive+scp.",
    )
    sync.set_defaults(func=command_sync)

    bootstrap = sub.add_parser("bootstrap", help="run remote dependency bootstrap")
    bootstrap.set_defaults(func=command_bootstrap)

    start = sub.add_parser("start", help="start one remote tmux reviewer 2x2 run")
    start.add_argument("--model", default="local_ds70b")
    start.add_argument("--crawler", default="default")
    start.add_argument("--session", default="ds70b_2x2")
    start.add_argument("--out-dir", default=None)
    start.add_argument("--samples", type=int, default=None)
    start.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional Hydra override appended to every 2x2 cell command.",
    )
    start.add_argument(
        "--env",
        action="append",
        default=[],
        help="Extra environment variable for the remote run, as NAME=VALUE.",
    )
    start.add_argument(
        "--validate-all-discovered",
        action="store_true",
        help="Enable full per-candidate refusal filtering in the 2x2 cells.",
    )
    start.set_defaults(func=command_start)

    status = sub.add_parser("status", help="show remote session and latest log tail")
    status.add_argument("--session", default="ds70b_2x2")
    status.set_defaults(func=command_status)

    stop = sub.add_parser("stop", help="stop a remote tmux reviewer run")
    stop.add_argument("--session", default="ds70b_2x2")
    stop.set_defaults(func=command_stop)

    tail = sub.add_parser("tail", help="follow the latest remote run.log")
    tail.add_argument("--session", default="ds70b_2x2")
    tail.add_argument("--lines", type=int, default=80)
    tail.set_defaults(func=command_tail)

    fetch = sub.add_parser("fetch", help="fetch latest or selected output directory")
    fetch.add_argument("--remote-out-dir", default=None)
    fetch.add_argument("--local-dir", default=None)
    fetch.add_argument(
        "--transfer-method",
        choices=("auto", "rsync", "archive"),
        default="auto",
        help="Transfer implementation. auto tries rsync then archive+scp.",
    )
    fetch.set_defaults(func=command_fetch)

    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
