"""Replicate the RPP text-prediction experiment with Qwen3.8-Flash on Together AI.

Same claims, same (unchanged) A0 prompt, same parser as
prediction/llm_pred/predict_text_batch.py; only the model and transport differ.

Two transports, same output CSV:
  --mode direct   concurrent synchronous calls (Together's chat template); good for pilots
  --mode submit   Together Batch API (up to 50% off serverless models, <24h); then
  --mode status / --mode collect

Re-running resumes: claim/run pairs that already have a valid q_hat are skipped.

    python ablation/qwen_experiments/run_qwen.py --dry-run --limit 3 --runs 2
    python ablation/qwen_experiments/run_qwen.py --limit 3 --runs 2                   # direct pilot
    python ablation/qwen_experiments/run_qwen.py --mode submit --runs 10              # batch
    python ablation/qwen_experiments/run_qwen.py --mode status  --runs 10
    python ablation/qwen_experiments/run_qwen.py --mode collect --runs 10

Needs the `together` package and TOGETHER_API_KEY (read from the repo-root .env).
"""
import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "prediction" / "llm_pred"))
sys.path.insert(0, str(HERE.parent / "action_intervention"))
import predict_text_batch as ptb  # noqa: E402  (also puts prompt/ on sys.path)
import run_ablation as ra  # noqa: E402  (prompt variants for --arm)

DEFAULT_MODEL = "Qwen/Qwen3.8-Flash"
# SHA-256 of the unchanged A0 system prompt (see ablation/action_intervention/PLAN.md).
A0_PROMPT_SHA256 = "9c0500d00349aa6f8639016efebf14215529e538c4464df9166cbf6af23acf70"
MAX_BATCH_FILE_BYTES = 90_000_000  # Together limit is 100 MB per input file
TERMINAL_FAILED = {"FAILED", "EXPIRED", "CANCELLED"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", choices=["direct", "submit", "status", "collect"], default="direct")
    p.add_argument("--arm", default="baseline", choices=["baseline", *sorted(ra.VARIANTS)],
                   help="Step-1 extraction prompt: baseline = unchanged A0 prompt; n_only / claim_only "
                        "= the ablation arms of action_intervention/PLAN.md")
    p.add_argument("--model", default=os.getenv("TOGETHER_MODEL", DEFAULT_MODEL))
    p.add_argument("--runs", type=int, default=100, help="runs per claim (A0 used 100)")
    p.add_argument("--limit", type=int, default=0, help="only the first N claims")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--reasoning-effort", default="high", choices=["low", "medium", "high", "default"],
                   help="sent explicitly as reasoning_effort (default: high); 'default' omits it and "
                        "lets Together pick (the earlier pilots did this; files are named *_default_*)")
    p.add_argument("--max-tokens", type=int, default=16384, help="same budget as gpt-5.4-mini 'high'")
    p.add_argument("--workers", type=int, default=8, help="direct mode: concurrent requests")
    p.add_argument("--max-retries", type=int, default=6, help="direct mode: retries per request")
    p.add_argument("--force", action="store_true",
                   help="submit even though earlier batches have not been collected")
    p.add_argument("--dry-run", action="store_true", help="build requests and report, no API calls")
    return p.parse_args()


# ── Paths, resume, records ───────────────────────────────────────────────────

def paths(args):
    slug = args.model.split("/")[-1]
    tag = f"{args.reasoning_effort}_temp{args.temperature}"
    out_dir = HERE / "results" / slug / ("RPP" if args.arm == "baseline" else args.arm)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"text_predictions_{tag}.csv", out_dir / f"batch_meta_{tag}.json", out_dir


def done_pairs(csv):
    if not csv.exists():
        return set()
    d = pd.read_csv(csv)
    d = d[d["q_hat"].notna()]
    return set(zip(d["altmejd_id"], d["run_id"].astype(int)))


def build_jobs(args, data, done):
    jobs, cache = [], {}
    for item in data["items"]:
        if item["title"] not in cache:
            cache[item["title"]] = ptb.read_paper("rpp", item, 0)
        paper = cache[item["title"]]
        if paper is None:
            print(f"Skipping (missing OCR text): {item['title']}")
            continue
        messages = ptb.messages_for("rpp", data, item, paper)
        for run_id in range(args.runs):
            if (item["altmejd_id"], run_id) not in done:
                jobs.append((item, run_id, messages))
    jobs.sort(key=lambda job: job[1])  # stable: round by round (run 0 for every claim, then run 1, ...)
    return jobs


def request_body(args, messages):
    body = dict(model=args.model, messages=messages,
                temperature=args.temperature, max_tokens=args.max_tokens)
    if args.reasoning_effort != "default":
        body["reasoning_effort"] = args.reasoning_effort
    return body


def make_record(args, item, run_id, content, finish_reason, prompt_tokens, completion_tokens):
    rec = ptb.response_record("rpp", item, run_id, content or "")
    rec.update(model=args.model, reasoning_effort=args.reasoning_effort,
               temperature=args.temperature, finish_reason=finish_reason,
               prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return rec


def append_rows(csv, rows):
    if rows:
        pd.DataFrame(rows).to_csv(csv, mode="a", header=not csv.exists(), index=False)


def finalize(csv):
    """Keep one row per claim/run (a retried pair can leave two), preferring a valid q_hat."""
    if not csv.exists():
        return
    d = pd.read_csv(csv)
    d = (d.assign(_ok=d["q_hat"].notna()).sort_values("_ok", kind="stable")
          .drop_duplicates(["altmejd_id", "run_id"], keep="last").drop(columns="_ok")
          .sort_values(["altmejd_id", "run_id"]))
    d.to_csv(csv, index=False)
    print(f"{csv.name}: {len(d)} rows, {d['altmejd_id'].nunique()} claims; "
          f"q_hat parsed {d['q_hat'].notna().mean():.1%}, verdict parsed "
          f"{(d['verdict'].fillna('') != '').mean():.1%}")


def client_or_exit():
    load_dotenv(ptb.REPO_ROOT / ".env")
    try:
        from together import Together
    except ImportError:
        raise SystemExit("The `together` package is not installed: uv pip install together")
    return Together()


# ── Direct mode ──────────────────────────────────────────────────────────────

def complete_rounds(csv, n_claims):
    """Number of leading run ids 0..R-1 that have a valid row for every claim (the usable common R)."""
    d = pd.read_csv(csv)
    d = d[d["q_hat"].notna()]
    per_run = d.groupby("run_id")["altmejd_id"].nunique()
    r = 0
    while per_run.get(r, 0) >= n_claims:
        r += 1
    return r


def run_direct(args, client, jobs, csv):
    def stream(messages):
        """Streamed call (the model rejects non-streaming); returns content, finish_reason, usage."""
        parts, finish, usage = [], None, None
        for chunk in client.chat.completions.create(**request_body(args, messages), stream=True):
            usage = getattr(chunk, "usage", None) or usage  # Together sends usage on the last chunk
            for ch in chunk.choices or []:
                delta = getattr(ch, "delta", None)
                if delta is not None and getattr(delta, "content", None):
                    parts.append(delta.content)  # final answer only; reasoning tokens are not kept
                finish = getattr(ch, "finish_reason", None) or finish
        return "".join(parts), finish, usage

    def call(messages):
        for attempt in range(args.max_retries + 1):
            try:  # the whole stream is retried, so a dropped connection mid-answer restarts it
                return stream(messages)
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                permanent = status is not None and 400 <= status < 500 and status not in (408, 409, 429)
                if permanent or attempt == args.max_retries:  # e.g. 403 permission: retrying cannot help
                    raise
                wait = min(60, 2 ** attempt) + random.random()
                print(f"  retry {attempt + 1}/{args.max_retries} in {wait:.0f}s: {type(exc).__name__}: {exc}")
                time.sleep(wait)

    def work(item, run_id, messages):
        content, finish, u = call(messages)
        return make_record(args, item, run_id, content, finish,
                           getattr(u, "prompt_tokens", None), getattr(u, "completion_tokens", None))

    n_claims = len({job[0]["altmejd_id"] for job in jobs})
    lock, ok, failed = threading.Lock(), 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, *job): job for job in jobs}
        for k, fut in enumerate(as_completed(futures), 1):
            item, run_id, _ = futures[fut]
            try:
                rec = fut.result()
            except Exception as exc:
                failed += 1
                print(f"FAILED {item['altmejd_id']} run {run_id}: {type(exc).__name__}: {exc}")
                continue
            with lock:
                append_rows(csv, [rec])
                ok += 1
            if k % 25 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)} done ({failed} failed); "
                      f"runs complete for all claims: {complete_rounds(csv, n_claims)}")
    print(f"Direct: {ok} saved, {failed} failed (rerun to retry).")
    finalize(csv)


# ── Batch mode ───────────────────────────────────────────────────────────────

def load_meta(meta_path):
    return json.loads(meta_path.read_text()) if meta_path.exists() else {"batches": []}


def save_meta(meta_path, meta):
    meta_path.write_text(json.dumps(meta, indent=2))


def batch_submit(args, client, jobs, meta_path):
    meta = load_meta(meta_path)
    pending = [b for b in meta["batches"] if not b.get("collected")]
    if pending and not args.force:
        raise SystemExit(f"{len(pending)} earlier batch(es) not collected yet; run --mode collect first "
                         "(or pass --force to submit anyway and risk paying twice).")
    lines = [json.dumps({"custom_id": f"{item['altmejd_id']}|{run_id}",
                         "body": request_body(args, messages)}) for item, run_id, messages in jobs]
    chunks, cur, size = [], [], 0
    for line in lines:
        n = len(line.encode("utf-8")) + 1
        if cur and size + n > MAX_BATCH_FILE_BYTES:
            chunks.append(cur)
            cur, size = [], 0
        cur.append(line)
        size += n
    if cur:
        chunks.append(cur)
    for k, chunk in enumerate(chunks, 1):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
            fh.write("\n".join(chunk) + "\n")
            path = fh.name
        try:
            file_resp = client.files.upload(file=path, purpose="batch-api", check=False)
        finally:
            os.unlink(path)
        batch = client.batches.create(input_file_id=file_resp.id, endpoint="/v1/chat/completions").job
        meta["batches"].append({"batch_id": batch.id, "file_id": file_resp.id, "requests": len(chunk),
                                "model": args.model, "runs": args.runs, "collected": False})
        save_meta(meta_path, meta)
        print(f"  [{k}/{len(chunks)}] {len(chunk)} requests → batch {batch.id} ({batch.status})")
    print(f"Meta → {meta_path}\nNext: --mode status, then --mode collect (same --runs/--temperature/"
          "--reasoning-effort so the paths match).")


def batch_status(client, meta_path):
    meta = load_meta(meta_path)
    if not meta["batches"]:
        raise SystemExit(f"No batches recorded in {meta_path}; run --mode submit first.")
    for b in meta["batches"]:
        batch = client.batches.retrieve(b["batch_id"])
        flag = " (collected)" if b.get("collected") else ""
        print(f"{b['batch_id']}: {batch.status}, progress={getattr(batch, 'progress', None)}, "
              f"{b['requests']} requests{flag}")


def download(client, file_id, dest):
    with client.files.with_streaming_response.content(id=file_id) as response:
        with open(dest, "wb") as f:
            for chunk in response.iter_bytes():
                f.write(chunk)


def batch_collect(args, client, data, csv, meta_path, out_dir):
    meta = load_meta(meta_path)
    lookup = {item["altmejd_id"]: item for item in data["items"]}
    for b in meta["batches"]:
        if b.get("collected"):
            continue
        batch = client.batches.retrieve(b["batch_id"])
        if batch.status != "COMPLETED":
            note = " — will not complete; resubmit the missing pairs" if batch.status in TERMINAL_FAILED else ""
            print(f"{b['batch_id']}: {batch.status}{note}")
            if batch.status in TERMINAL_FAILED:
                b["collected"] = True  # nothing to download; a new submit picks up the missing pairs
                save_meta(meta_path, meta)
            continue
        rows, errors = [], 0
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "out.jsonl"
            download(client, batch.output_file_id, out_path)
            for line in out_path.read_text().splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                claim_id, run_id = obj["custom_id"].rsplit("|", 1)
                resp = obj.get("response") or {}
                body = resp.get("body") or {}
                if resp.get("status_code") != 200 or not body.get("choices"):
                    errors += 1
                    continue
                choice, usage = body["choices"][0], body.get("usage") or {}
                rows.append(make_record(args, lookup[claim_id], int(run_id),
                                        (choice.get("message") or {}).get("content"),
                                        choice.get("finish_reason"),
                                        usage.get("prompt_tokens"), usage.get("completion_tokens")))
        error_file = getattr(batch, "error_file_id", None)
        if error_file:
            dest = out_dir / f"batch_errors_{b['batch_id']}.jsonl"
            download(client, error_file, dest)
            print(f"  error file → {dest}")
        append_rows(csv, rows)
        b["collected"] = True
        save_meta(meta_path, meta)
        print(f"{b['batch_id']}: {len(rows)} rows saved, {errors} errored lines")
    finalize(csv)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if args.arm != "baseline":
        ptb.prompt_text.SYSTEM_PROMPT = ra.build_prompt(args.arm)  # read by messages_for at call time
    sha = hashlib.sha256(ptb.prompt_text.SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    if args.arm == "baseline" and sha != A0_PROMPT_SHA256:
        raise SystemExit(f"System prompt sha256 {sha} != A0 {A0_PROMPT_SHA256}; prompt has changed.")
    print(f"[run_qwen:{args.mode}] arm={args.arm} model={args.model} effort={args.reasoning_effort} "
          f"temp={args.temperature} prompt sha256={sha[:12]}…"
          + (" (matches A0)" if args.arm == "baseline" else ""))

    data = ptb.load_corpus("rpp", args.limit)
    csv, meta_path, out_dir = paths(args)
    if args.mode in ("direct", "submit"):
        (out_dir / "system_prompt.txt").write_text(ptb.prompt_text.SYSTEM_PROMPT)

    if args.mode == "status":
        return batch_status(client_or_exit(), meta_path)
    if args.mode == "collect":
        return batch_collect(args, client_or_exit(), data, csv, meta_path, out_dir)

    done = done_pairs(csv)
    if done:
        print(f"Resuming: {len(done)} valid claim/run pairs already in {csv.name}")
    jobs = build_jobs(args, data, done)
    n_chars = sum(sum(len(m["content"]) for m in msgs) for _, _, msgs in jobs)
    print(f"{len(jobs)} requests to send ({len(data['items'])} claims × {args.runs} runs, minus done); "
          f"~{n_chars / 4 / 1e6:.2f}M input tokens → {csv}")
    if args.dry_run or not jobs:
        return
    client = client_or_exit()
    if args.mode == "direct":
        run_direct(args, client, jobs, csv)
    else:
        batch_submit(args, client, jobs, meta_path)


if __name__ == "__main__":
    main()
