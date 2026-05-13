"""Extract Sotopia's 90 environment profiles + agent profiles from the official
dump.rdb snapshot, without running `sotopia install`.

Strategy: download dump.rdb if missing, then spin up redis/redis-stack-server in a
short-lived Docker container with that file mounted as the persistence file. Once
Redis loads the snapshot, scan all redis-om JsonModel keys, JSON.GET each value,
and write JSONL files. The container is removed at the end.

Usage:
    python scripts/extract_sotopia_seeds.py
    python scripts/extract_sotopia_seeds.py --dataset sotopia-pi   # larger dataset

Requirements: Docker running locally, redis (python) installed.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import redis


# NOTE on dataset choice:
# The original ICLR 2024 "Sotopia Dataset" URL (CMU Box) listed in sotopia's
# published_datasets.json is currently dead (HTTP 403). The cmu-lti/sotopia HF repo
# only hosts episode CSVs, not the structured EnvironmentProfile/AgentProfile data.
# The cmu-lti/sotopia-pi dump.rdb is the working source and is a *superset* of the
# original 90 scenarios with the same schema, so we default to it.
DATASETS = {
    "sotopia-pi": "https://huggingface.co/datasets/cmu-lti/sotopia-pi/resolve/main/dump.rdb?download=true",
    "sotopia": "https://cmu.box.com/shared/static/xiivc5z8rnmi1zr6vmk1ohxslylvynur",  # currently 403
    "agent_vs_script": "https://huggingface.co/datasets/cmu-lti/agent_vs_script/resolve/main/dump.rdb?download=true",
}

# Each redis-om JsonModel key in the snapshot looks like:
#   :sotopia.database.persistent_profile.EnvironmentProfile:<pk>
# We pull these models. Add more here if you want EpisodeLog etc.
TARGETS = {
    "environment_profiles.jsonl":
        ":sotopia.database.persistent_profile.EnvironmentProfile:*",
    "agent_profiles.jsonl":
        ":sotopia.database.persistent_profile.AgentProfile:*",
    "relationship_profiles.jsonl":
        ":sotopia.database.persistent_profile.RelationshipProfile:*",
    "environment_lists.jsonl":
        ":sotopia.database.persistent_profile.EnvironmentList:*",
}

CONTAINER_NAME = "sotopia-seed-extractor"
HOST_PORT = 16379  # avoid clashing with a local 6379


def download_rdb(url: str, target: Path) -> None:
    if target.exists():
        print(f"[skip] {target} already exists ({target.stat().st_size / 1e6:.1f} MB)")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if not shutil.which("curl"):
        raise RuntimeError("curl is required to download the RDB but was not found in PATH.")
    print(f"[download] {url}\n           -> {target}")
    # Match the official sotopia install behavior (curl follows Box redirects cleanly).
    subprocess.run(["curl", "-L", "--fail", "-o", str(target), url], check=True)
    print(f"[ok] downloaded {target.stat().st_size / 1e6:.1f} MB")


def run(cmd: list[str], check: bool = True, **kw) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, check=False, capture_output=True, text=True, **kw)
    if check and result.returncode != 0:
        msg = (f"Command {cmd!r} failed (exit {result.returncode})\n"
               f"  stdout: {result.stdout.strip()}\n  stderr: {result.stderr.strip()}")
        raise RuntimeError(msg)
    return result


def ensure_docker_daemon() -> None:
    r = subprocess.run(["docker", "info"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            "Docker daemon is not running. On macOS, open Docker Desktop and wait "
            "for it to finish starting, then rerun this script.\n"
            f"docker info stderr: {r.stderr.strip()}"
        )


def stop_container() -> None:
    run(["docker", "rm", "-f", CONTAINER_NAME], check=False)


def start_redis_stack(rdb_dir: Path) -> None:
    """Start redis-stack-server with the dump.rdb mounted.

    redis-stack-server reads /data/dump.rdb on boot, so we mount the
    directory containing dump.rdb at /data.
    """
    stop_container()
    print(f"[docker] starting redis-stack-server with data dir {rdb_dir}")
    run([
        "docker", "run", "-d", "--rm",
        "--name", CONTAINER_NAME,
        "-p", f"{HOST_PORT}:6379",
        "-v", f"{rdb_dir.resolve()}:/data",
        "redis/redis-stack-server:latest",
    ])


def wait_for_redis(timeout: float = 60.0) -> redis.Redis:
    deadline = time.time() + timeout
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            r = redis.Redis(host="localhost", port=HOST_PORT, decode_responses=True)
            if r.ping():
                modules = set()
                for m in r.module_list():
                    name = m.get("name") if isinstance(m, dict) else None
                    if name is None and isinstance(m, dict):
                        name = m.get(b"name")
                    if isinstance(name, bytes):
                        name = name.decode()
                    if name:
                        modules.add(name.lower())
                if "rejson" in modules or "json" in modules:
                    return r
        except Exception as e:
            last_err = e
        time.sleep(0.5)
    raise RuntimeError(f"Redis did not become ready in {timeout}s: {last_err}")


def export_keys(r: redis.Redis, pattern: str, out_path: Path) -> int:
    n = 0
    with open(out_path, "w") as f:
        for key in r.scan_iter(match=pattern, count=500):
            try:
                value = r.execute_command("JSON.GET", key)
            except redis.ResponseError as e:
                print(f"  [warn] JSON.GET failed for {key}: {e}")
                continue
            if value is None:
                continue
            # value comes back as a JSON string; round-trip to normalize
            try:
                obj = json.loads(value)
            except json.JSONDecodeError:
                obj = {"raw": value}
            # Stamp the redis-om primary key so downstream code can dedupe / cross-reference.
            pk = key.rsplit(":", 1)[-1]
            obj.setdefault("pk", pk)
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS.keys()), default="sotopia-pi",
                    help="Which published dump.rdb to extract (default: sotopia, the ICLR 90 scenarios)")
    ap.add_argument("--data-dir", default="data/sotopia_seeds",
                    help="Where to place dump.rdb and write the JSONL files")
    ap.add_argument("--keep-container", action="store_true",
                    help="Leave the redis container running for inspection")
    args = ap.parse_args()

    if not shutil.which("docker"):
        print("ERROR: docker is required but not found in PATH.", file=sys.stderr)
        return 2
    try:
        ensure_docker_daemon()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3

    data_dir = Path(args.data_dir)
    rdb_path = data_dir / "dump.rdb"
    download_rdb(DATASETS[args.dataset], rdb_path)

    try:
        start_redis_stack(data_dir)
        r = wait_for_redis()
        print(f"[ok] redis ready on localhost:{HOST_PORT}")

        totals = {}
        for filename, pattern in TARGETS.items():
            out_path = data_dir / filename
            count = export_keys(r, pattern, out_path)
            totals[filename] = count
            print(f"[export] {filename}: {count} records")
            if count == 0:
                out_path.unlink(missing_ok=True)

        print("\n=== Summary ===")
        for k, v in totals.items():
            print(f"  {k}: {v}")

        # Quick sample
        env_path = data_dir / "environment_profiles.jsonl"
        if env_path.exists():
            with open(env_path) as f:
                sample = json.loads(f.readline())
            print("\nSample environment_profiles[0] keys:", sorted(sample.keys()))
            preview = (sample.get("scenario") or "")[:120].replace("\n", " ")
            print(f"  scenario preview: {preview!r}")
    finally:
        if not args.keep_container:
            stop_container()
            print("[docker] container removed")
        else:
            print(f"[docker] container '{CONTAINER_NAME}' left running on port {HOST_PORT}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
