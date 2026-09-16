"""Contamination probe: query gpt-5.4-mini for verbatim recall of a study's
replication outcome from its title plus a one-line locator for the specific
finding being asked about (all three corpora -- see prompt_contamination.py's
module docstring for each corpus's locator field, and for why CB specifically
needs effect-level, not paper-level, granularity: 17/23 CB papers mix
replicable and unreplicable outcomes across their own effects). No paper
text is included anywhere in this script -- a correct, confident answer
indicates the model already "knows" the outcome from training data, not from
reasoning over text the main pipeline feeds it.

One corpus-agnostic script (unlike predict_text_batch{,_cb,_ssrp}.py, which
fork three ways to handle each corpus's own paper-text format) -- this probe
needs no paper-text handling at all, so there's no reason to fork it.

Fixed at this project's canonical config (reasoning_effort=high, temp=0.7)
and a light R=10 repeated queries per unit -- this is a robustness/validity
check, not a headline result, so it doesn't need the main pipeline's full
effort x temp grid or R=100. (REASONING_EFFORT is still read from the env,
same knob predict_text_batch*.py exposes, in case a future run wants to
compare effort levels; temp is not parameterized here at all since gpt-5.4-mini
ignores it as a reasoning model, same as elsewhere in this repo.)

Three phases via --mode, same submit/status/collect shape as
predict_text_batch.py --corpus cb, including its custom_id-assigned-before-chunking
and skip/resume-on-valid-rows fixes:

  submit   Build .jsonl request file(s), upload, create the batch(es).
  status   Print current state of all submitted batches (or one, with --batch-id).
  collect  Once every batch is "completed", download, parse, write one CSV.

Run from repo root:
    python prediction/llm_pred/predict_contamination_probe.py --corpus RPP --mode submit --runs 10
    python prediction/llm_pred/predict_contamination_probe.py --corpus RPP --mode status
    python prediction/llm_pred/predict_contamination_probe.py --corpus RPP --mode collect
    (repeat --corpus CB, --corpus SSRP)

Writes: prediction/LLM_Reasoning/{RPP,CB,SSRP}/contamination_probe_high_temp0.7.csv
"""
import argparse
import json
import os
import re
import sys
import tempfile

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "prompt"))
sys.path.insert(0, os.path.join(REPO, "data"))
import prompt_contamination as pc  # noqa: E402

load_dotenv()

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--corpus",  required=True, choices=["RPP", "CB", "SSRP"])
parser.add_argument("--mode",    required=True, choices=["submit", "status", "collect"])
parser.add_argument("--runs",    type=int, default=10, help="repeated queries per unit (submit only)")
parser.add_argument("--batch-id", default=None, help="override batch_id (otherwise read from meta file)")
args = parser.parse_args()

# ── Provider / model ──────────────────────────────────────────────────────────
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openai")
if MODEL_PROVIDER != "openai":
    sys.exit("Batch API is only supported for MODEL_PROVIDER=openai.")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), max_retries=8, timeout=120.0)
MODEL = os.getenv("MODEL", "gpt-4.1-nano")

_IS_REASONING = MODEL.lower().startswith(("gpt-5", "o1", "o3", "o4"))
_REASONING_BUDGET = {"low": 4096, "medium": 8192, "high": 16384}
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "high")

if _IS_REASONING:
    BASE_BODY = {
        "model": MODEL,
        "max_completion_tokens": _REASONING_BUDGET.get(REASONING_EFFORT, 4096),
        "reasoning_effort": REASONING_EFFORT,
    }
else:
    BASE_BODY = {"model": MODEL, "max_tokens": 256, "temperature": 0.7}

# ── Paths ─────────────────────────────────────────────────────────────────────
_effort_tag = REASONING_EFFORT if _IS_REASONING else "openai"
OUT_DIR = os.path.join(REPO, "prediction", "LLM_Reasoning", args.corpus)
OUT_CSV = os.path.join(OUT_DIR, f"contamination_probe_{_effort_tag}_temp0.7.csv")
META_FILE = os.path.join(OUT_DIR, f"contamination_probe_batch_meta_{_effort_tag}_temp0.7.json")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────
UNIT_KEY_COLS = {"RPP": ["unit_id"], "CB": ["paper_num", "experiment_num", "effect_num"],
                  "SSRP": ["study_num"]}
UNITS = {"RPP": pc.load_rpp_units, "CB": pc.load_cb_units, "SSRP": pc.load_ssrp_units}[args.corpus]()
KEY_COLS = UNIT_KEY_COLS[args.corpus]


def _json_safe(v):
    """numpy int64/float64 (from pandas) aren't JSON-serializable -- .item()
    converts to native Python types; plain str/already-native values pass
    through unchanged. Same fix predict_text_batch.py --corpus cb applies via
    explicit int(...) casts, generalized here since KEY_COLS types vary
    (RPP's unit_id is a string, CB/SSRP's are ints)."""
    return v.item() if hasattr(v, "item") else v


def unit_row(key_tuple):
    mask = pd.Series(True, index=UNITS.index)
    for col, val in zip(KEY_COLS, key_tuple):
        mask &= (UNITS[col] == val)
    return UNITS[mask].iloc[0]


# ── Parsers ───────────────────────────────────────────────────────────────────
def parse_familiarity(text):
    m = re.search(r"FAMILIARITY:\s*(recognized|unsure|unfamiliar)", text, re.IGNORECASE)
    return m.group(1).lower() if m else ""


def parse_verdict(text):
    m = re.search(r"VERDICT:\s*(replicable|unreplicable|unknown)", text, re.IGNORECASE)
    return m.group(1).lower() if m else ""


def parse_notes(text):
    m = re.search(r"NOTES:\s*(.+)", text, re.IGNORECASE)
    return m.group(1).strip() if m else ""


def parse_response(text, key_tuple, run_id):
    row = unit_row(key_tuple)
    rec = dict(zip(KEY_COLS, key_tuple))
    rec.update({
        "run_id": run_id,
        "title": row["title"],
        "familiarity": parse_familiarity(text),
        "verdict": parse_verdict(text),
        "notes": parse_notes(text),
        "reasoning": text,
        "ground_truth": row["ground_truth"],
    })
    return rec


# Kept for safety, matching predict_text_batch.py --corpus cb's fix for the same
# 200MB-per-batch-file cap -- unlikely to ever trigger here since these
# requests carry no paper text (~880/1580/210 tiny requests per corpus).
MAX_CHUNK_BYTES = int(os.getenv("MAX_CHUNK_BYTES", 180_000_000))


def _chunk_requests(index, requests):
    chunk_index, chunk_reqs, chunk_bytes = [], [], 0
    for idx_entry, req in zip(index, requests):
        line_bytes = len(json.dumps(req).encode("utf-8")) + 1
        if chunk_reqs and chunk_bytes + line_bytes > MAX_CHUNK_BYTES:
            yield chunk_index, chunk_reqs
            chunk_index, chunk_reqs, chunk_bytes = [], [], 0
        chunk_index.append(idx_entry)
        chunk_reqs.append(req)
        chunk_bytes += line_bytes
    if chunk_reqs:
        yield chunk_index, chunk_reqs


# ── Mode: submit ──────────────────────────────────────────────────────────────
def do_submit():
    done = set()
    if os.path.exists(OUT_CSV):
        try:
            ex = pd.read_csv(OUT_CSV)
            if {"run_id", "verdict"} <= set(ex.columns):
                valid = ex[ex["verdict"].notna() & (ex["verdict"] != "")]
                done = set(zip(*[valid[c].tolist() for c in KEY_COLS],
                                valid["run_id"].astype(int).tolist()))
                print(f"Resuming: {len(done)} valid (unit, run) pairs already done — skipping.")
        except pd.errors.EmptyDataError:
            pass

    # custom_id assigned from a GLOBAL running count here, BEFORE
    # _chunk_requests() splits requests across batches -- the exact bug
    # predict_text_batch{,_cb}.py's do_submit hit and fixed twice this
    # session, avoided from the start here.
    index = []   # index[i] = [custom_id, *key_values, run_id]
    requests = []
    for _, row in tqdm(UNITS.iterrows(), total=len(UNITS), desc="Building requests"):
        key_tuple = tuple(_json_safe(row[c]) for c in KEY_COLS)
        messages = pc.build_messages(row["title"], row["locator"])
        for run_id in range(args.runs):
            if (*key_tuple, run_id) in done:
                continue
            custom_id = f"r{len(index):06d}"
            index.append([custom_id, *key_tuple, run_id])
            requests.append({
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {**BASE_BODY, "messages": messages},
            })

    if not requests:
        print("Nothing to submit — all runs already done.")
        return

    chunks = list(_chunk_requests(index, requests))
    print(f"Submitting {len(requests)} requests for {MODEL} ({args.corpus}), split into "
          f"{len(chunks)} batch(es)...")

    batches_meta = []
    for i, (chunk_index, chunk_reqs) in enumerate(chunks):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for req in chunk_reqs:
                f.write(json.dumps(req) + "\n")
            tmp_path = f.name
        size_mb = os.path.getsize(tmp_path) / 1024 / 1024

        with open(tmp_path, "rb") as f:
            uploaded = client.files.create(file=f, purpose="batch")
        os.unlink(tmp_path)
        print(f"  [{i+1}/{len(chunks)}] {len(chunk_reqs)} requests, {size_mb:.1f}MB "
              f"→ uploaded {uploaded.id}")

        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"model": MODEL, "corpus": args.corpus, "runs": str(args.runs),
                      "chunk": f"{i+1}/{len(chunks)}"},
        )
        print(f"  [{i+1}/{len(chunks)}] batch {batch.id}  status={batch.status}")
        batches_meta.append({"batch_id": batch.id, "file_id": uploaded.id, "index": chunk_index})

    meta = {"model": MODEL, "corpus": args.corpus, "runs": args.runs, "batches": batches_meta}
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSubmitted {len(batches_meta)} batch(es), {len(requests)} requests total.")
    print(f"Meta saved → {META_FILE}")
    print(f"\nCheck status:   python prediction/llm_pred/predict_contamination_probe.py --corpus {args.corpus} --mode status")
    print(f"Collect results: python prediction/llm_pred/predict_contamination_probe.py --corpus {args.corpus} --mode collect")


# ── Mode: status ──────────────────────────────────────────────────────────────
def do_status():
    if args.batch_id:
        batch_ids = [args.batch_id]
    else:
        meta = _load_meta()
        batch_ids = [b["batch_id"] for b in meta["batches"]]

    tot_total = tot_completed = tot_failed = 0
    for batch_id in batch_ids:
        batch = client.batches.retrieve(batch_id)
        c = batch.request_counts
        print(f"Batch:     {batch_id}")
        print(f"Status:    {batch.status}")
        print(f"Requests:  total={c.total}  completed={c.completed}  failed={c.failed}")
        if batch.output_file_id:
            print(f"Output:    {batch.output_file_id}")
        if batch.error_file_id:
            print(f"Errors:    {batch.error_file_id}")
        print()
        tot_total += c.total
        tot_completed += c.completed
        tot_failed += c.failed

    if len(batch_ids) > 1:
        print(f"TOTAL across {len(batch_ids)} batches: "
              f"total={tot_total}  completed={tot_completed}  failed={tot_failed}")


# ── Mode: collect ─────────────────────────────────────────────────────────────
def do_collect():
    meta = _load_meta()
    batches = ([{"batch_id": args.batch_id,
                 "index": next(b["index"] for b in meta["batches"] if b["batch_id"] == args.batch_id)}]
               if args.batch_id else meta["batches"])

    not_done = []
    for b in batches:
        status = client.batches.retrieve(b["batch_id"]).status
        if status != "completed":
            not_done.append((b["batch_id"], status))
    if not_done:
        print("Not all batches are complete yet:")
        for bid, status in not_done:
            print(f"  {bid}: {status}")
        sys.exit(1)

    existing = pd.DataFrame()
    if os.path.exists(OUT_CSV):
        try:
            existing = pd.read_csv(OUT_CSV)
            existing = existing[existing["verdict"].notna() & (existing["verdict"] != "")]
        except pd.errors.EmptyDataError:
            pass

    records = []
    errors = 0
    for b in batches:
        index = {e[0]: tuple(e[1:]) for e in b["index"]}  # custom_id -> (*key_values, run_id)

        batch = client.batches.retrieve(b["batch_id"])
        print(f"Downloading output for {b['batch_id']} ({batch.output_file_id})...")
        content = client.files.content(batch.output_file_id).text

        for line in content.strip().split("\n"):
            if not line.strip():
                continue
            obj = json.loads(line)
            cid = obj["custom_id"]
            if cid not in index:
                print(f"Unknown custom_id: {cid}")
                continue
            *key_tuple, run_id = index[cid]

            if obj.get("error"):
                print(f"Error for {cid} ({args.corpus} {key_tuple} run {run_id}): {obj['error']}")
                errors += 1
                continue

            text = obj["response"]["body"]["choices"][0]["message"]["content"] or ""
            records.append(parse_response(text, tuple(key_tuple), run_id))

        if batch.error_file_id:
            errs = client.files.content(batch.error_file_id).text
            err_path = f"{OUT_DIR}/contamination_probe_batch_errors_{b['batch_id']}.jsonl"
            with open(err_path, "w") as f:
                f.write(errs)
            print(f"Error details → {err_path}")

    df = pd.concat([existing, pd.DataFrame(records)], ignore_index=True)
    df = df.drop_duplicates(subset=[*KEY_COLS, "run_id"], keep="last").reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(df)} rows → {OUT_CSV}  ({errors} errors)")

    if len(df):
        definite = df[df["verdict"] != "unknown"]
        print(f"Coverage (definite verdict): {len(definite)}/{len(df)} = {len(definite)/len(df):.3f}")
        if len(definite):
            acc = (definite["verdict"].map({"replicable": "yes", "unreplicable": "no"})
                   == definite["ground_truth"]).mean()
            print(f"Recall accuracy (definite verdicts only): {acc:.3f}")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _load_meta():
    if not os.path.exists(META_FILE):
        sys.exit(f"No meta file found at {META_FILE}. Run --mode submit first.")
    with open(META_FILE) as f:
        return json.load(f)


# ── Dispatch ──────────────────────────────────────────────────────────────────
print(f"[predict_contamination_probe] corpus={args.corpus}  model={MODEL}  mode={args.mode}  "
      f"{'runs=' + str(args.runs) + '  ' if args.mode == 'submit' else ''}out_dir={OUT_DIR}")

if args.mode == "submit":
    do_submit()
elif args.mode == "status":
    do_status()
elif args.mode == "collect":
    do_collect()
