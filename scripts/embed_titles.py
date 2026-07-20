#!/usr/bin/env python3
"""One-time: embed the abstract SCENARIO_TITLE (`social_dynamic | target_perspective`) for every
task in the gen90 archive, and cache to results/analysis/title_embeddings.json.

The archive's stored `embedding` is the *surface* embedding (full scenario text, deliberately
title-free). This produces the complementary *structural* embedding used by the structural-space
UMAP (scripts/plot_scenario_space.py). Cached, so it is computed once.

Run:  python3 scripts/embed_titles.py
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = ROOT / "results/gen90_expel/archive_latest.json"
OUT = ROOT / "results/analysis/title_embeddings.json"
EMBED_MODEL = "text-embedding-3-small"


def load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def title_text(s: dict) -> str:
    """The abstract structural key: 'social_dynamic | target_perspective'."""
    sd = (s.get("social_dynamic") or "").strip()
    tp = (s.get("target_perspective") or "").strip()
    if sd and tp:
        return f"{sd} | {tp}"
    return (s.get("scenario_title") or sd or tp or s.get("scenario", "")[:200]).strip()


def main():
    load_env()
    import openai

    tasks = json.load(open(ARCHIVE))["tasks"]
    ids = [t["id"] for t in tasks]
    texts = [title_text(t) for t in tasks]

    if OUT.exists():
        cache = json.load(open(OUT))
        if cache.get("model") == EMBED_MODEL and cache.get("ids") == ids:
            print(f"cache hit: {OUT} ({len(ids)} titles) — nothing to do")
            return

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    vecs = []
    B = 256
    for i in range(0, len(texts), B):
        chunk = texts[i:i + B]
        resp = client.embeddings.create(input=chunk, model=EMBED_MODEL)
        vecs.extend(d.embedding for d in resp.data)
        print(f"  embedded {min(i+B, len(texts))}/{len(texts)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"model": EMBED_MODEL, "ids": ids, "titles": texts, "embeddings": vecs},
              open(OUT, "w"))
    print(f"wrote {OUT}  ({len(vecs)} x {len(vecs[0])})")


if __name__ == "__main__":
    main()
