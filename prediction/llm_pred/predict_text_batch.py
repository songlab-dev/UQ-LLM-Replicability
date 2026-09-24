"""Submit, inspect, and collect text-prediction batches for RPP, CB, or SSRP.

Prompts come from prompt/prompt_text*.py, which read the anonymized paper
texts from file/, file_CB/, and file_SSRP/ under the repository root. The
source papers are not included in this repository for copyright reasons.

Run from any directory::

    MODEL=gpt-5.4-mini REASONING_EFFORT=high \
      python prediction/llm_pred/predict_text_batch.py --corpus rpp --mode submit --runs 20
    python prediction/llm_pred/predict_text_batch.py --corpus cb --mode status
    python prediction/llm_pred/predict_text_batch.py --corpus ssrp --mode collect
"""

import argparse
import json
import os
from pathlib import Path
import re
import sys
import tempfile

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

PACKAGE_DIR = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = PACKAGE_DIR.parent
sys.path.insert(0, str(PACKAGE_DIR))
sys.path.insert(0, str(PACKAGE_DIR / "prompt"))
sys.path.insert(0, str(PACKAGE_DIR / "data"))

import prompt_text  # noqa: E402
import prompt_text_cb  # noqa: E402
import prompt_text_ssrp  # noqa: E402
from rpp_outcomes import canonical_rpp_outcomes_for_titles, normalize_title  # noqa: E402


DEFAULT_RUNS = {"rpp": 20, "cb": 25, "ssrp": 10}
CORPUS_DIR = {"rpp": "RPP", "cb": "CB", "ssrp": "SSRP"}
DEDUP_COLUMNS = {
    "rpp": ["altmejd_id", "run_id"],
    "cb": ["paper_num", "experiment_num", "effect_num", "run_id"],
    "ssrp": ["study_num", "run_id"],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=sorted(DEFAULT_RUNS), default="rpp")
    parser.add_argument("--mode", required=True, choices=["submit", "status", "collect"])
    parser.add_argument("--runs", type=int, default=None, help="Monte-Carlo runs per claim")
    parser.add_argument("--limit", type=int, default=0, help="process only first N claims")
    parser.add_argument("--max-chars", type=int, default=0, help="truncate paper text")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--batch-id", default=None, help="override the batch ID in metadata")
    return parser.parse_args()


def native(value):
    return value.item() if isinstance(value, np.generic) else value


def load_corpus(corpus, limit):
    data_dir = PACKAGE_DIR / "data"
    if corpus == "rpp":
        frame = pd.read_csv(data_dir / "rpp_data_cleaned.csv", encoding="latin1")
        frame["_norm_title"] = frame["Study Title (O)"].map(normalize_title)
        if frame["_norm_title"].duplicated().any():
            raise ValueError("Duplicate normalized titles in rpp_data_cleaned.csv")

        alt = pd.read_csv(data_dir / "altmejd_osf" / "data.csv")
        claims = alt[(alt["project"] == "rpp") & (alt["drop"] == False)].copy()  # noqa: E712
        claims["_norm_title"] = claims["title"].map(normalize_title)
        claims = claims.merge(
            frame[["Study Title (O)", "_norm_title"]], on="_norm_title", how="inner"
        )
        if len(claims) != len(alt[(alt["project"] == "rpp") & (alt["drop"] == False)]):  # noqa: E712
            raise ValueError("Altmejd/RPP title merge dropped rows")
        if limit:
            claims = claims.head(limit)

        outcomes = canonical_rpp_outcomes_for_titles(claims["Study Title (O)"])
        if outcomes.isna().any():
            raise ValueError("Canonical RPP outcome missing for one or more claims")
        items = []
        for (_, claim), outcome in zip(claims.iterrows(), outcomes):
            title = claim["Study Title (O)"]
            source = frame[frame["Study Title (O)"] == title].iloc[0]
            items.append({
                "altmejd_id": native(claim["id"]),
                "title": title,
                "study_number": native(source["Replicated study number (R)"]),
                "ground_truth": "yes" if int(outcome) == 1 else "no",
            })
        return {"frame": frame, "items": items}

    if corpus == "cb":
        frame = pd.read_csv(data_dir / "cb_data_cleaned.csv")
        frame = frame.sort_values(["Paper #", "Experiment #", "Effect #"]).reset_index(drop=True)
        if limit:
            frame = frame.head(limit)
        return {"frame": frame, "items": [row for _, row in frame.iterrows()]}

    frame = pd.read_csv(data_dir / "ssrp_data_cleaned.csv")
    if limit:
        frame = frame.head(limit)
    return {"frame": frame, "items": [row for _, row in frame.iterrows()]}


def identity_fields(corpus, item):
    if corpus == "rpp":
        return (native(item["altmejd_id"]), str(item["title"]))
    if corpus == "cb":
        return (
            int(item["Paper #"]), int(item["Experiment #"]), int(item["Effect #"]),
        )
    return (int(item["Study Num"]),)


def resume_key(corpus, item, run_id):
    fields = identity_fields(corpus, item)
    return ((fields[0], run_id) if corpus == "rpp" else (*fields, run_id))


def read_paper(corpus, item, max_chars):
    if corpus == "rpp":
        return prompt_text.read_paper_text(item["title"], max_chars)
    if corpus == "cb":
        return prompt_text_cb.read_paper_text(item["dir"], max_chars)
    return prompt_text_ssrp.read_paper_text(int(item["Study Num"]), max_chars)


def messages_for(corpus, data, item, paper):
    if corpus == "rpp":
        return prompt_text.build_messages(
            data["frame"], item["title"], item["study_number"], paper
        )
    if corpus == "cb":
        return prompt_text_cb.build_messages(
            item["Study Title (O)"], item["Paper #"], item["Experiment #"],
            item["Effect #"], item["Description of effect (O)"], paper,
        )
    return prompt_text_ssrp.build_messages(data["frame"], int(item["Study Num"]), paper)


def item_label(corpus, item):
    if corpus == "rpp":
        return item["title"]
    return str(item["Study Title (O)"])


def parse_num(text, key):
    match = re.search(
        rf"{key}:\s*([<>]?\s*[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)",
        text,
        re.IGNORECASE,
    )
    if not match:
        return np.nan
    try:
        return float(match.group(1).replace("<", "").replace(">", "").strip())
    except ValueError:
        return np.nan


def parse_verdict(text):
    match = re.search(r"VERDICT:\s*(un)?replicable", text, re.IGNORECASE)
    return ("unreplicable" if match.group(1) else "replicable") if match else ""


def parse_choice(text, key, choices):
    match = re.search(rf"{key}:\s*(\w+)", text, re.IGNORECASE)
    value = match.group(1).strip().lower() if match else ""
    return value if value in choices else ""


def parse_qrp_severity(text):
    match = re.search(r"QRP_SEVERITY:\s*([0-3])", text, re.IGNORECASE)
    return int(match.group(1)) if match else np.nan


def response_record(corpus, item, run_id, text):
    verdict = parse_verdict(text)
    record = {
        "run_id": run_id,
        "prediction": verdict,
        "extract_N": parse_num(text, "EXTRACT_N"),
        "extract_p": parse_num(text, "EXTRACT_P"),
        "extract_r": parse_num(text, "EXTRACT_R"),
        "effect_type": parse_choice(text, "EFFECT_TYPE", {"main", "interaction", "other"}),
        "stat_score": parse_num(text, "STAT_SCORE"),
        "qrp_severity": parse_qrp_severity(text),
        "surprise_rating": parse_num(text, "SURPRISE_RATING"),
        "context": parse_choice(text, "CONTEXT_FRAGILITY", {"low", "medium", "high"}),
        "q_hat": parse_num(text, "Q_HAT"),
        "verdict": verdict,
        "reasoning": text,
    }
    if corpus == "rpp":
        return {
            "altmejd_id": item["altmejd_id"],
            "Study Title (O)": item["title"],
            "Study Number (R)": item["study_number"],
            **record,
            "ground_truth": item["ground_truth"],
        }
    if corpus == "cb":
        return {
            "paper_num": int(item["Paper #"]),
            "title": item["Study Title (O)"],
            "experiment_num": int(item["Experiment #"]),
            "effect_num": int(item["Effect #"]),
            **record,
            "ground_truth": str(item["Replicate (R)"]).strip().lower(),
        }
    return {
        "study_num": int(item["Study Num"]),
        "title": item["Study Title (O)"],
        **record,
        "ground_truth": str(item["Replicate (R)"]).strip().lower(),
    }


def chunk_requests(index, requests, max_bytes):
    chunk_index, chunk_requests_, chunk_bytes = [], [], 0
    for index_entry, request in zip(index, requests):
        line_bytes = len(json.dumps(request).encode("utf-8")) + 1
        if chunk_requests_ and chunk_bytes + line_bytes > max_bytes:
            yield chunk_index, chunk_requests_
            chunk_index, chunk_requests_, chunk_bytes = [], [], 0
        chunk_index.append(index_entry)
        chunk_requests_.append(request)
        chunk_bytes += line_bytes
    if chunk_requests_:
        yield chunk_index, chunk_requests_


class BatchJob:
    def __init__(self, args):
        self.args = args
        self.corpus = args.corpus
        self.runs = args.runs if args.runs is not None else DEFAULT_RUNS[self.corpus]
        self.data = load_corpus(self.corpus, args.limit)
        self.lookup = {
            tuple(identity_fields(self.corpus, item)): item for item in self.data["items"]
        }

        load_dotenv(REPO_ROOT / ".env")
        if os.getenv("MODEL_PROVIDER", "openai") != "openai":
            raise SystemExit("Batch API is only supported for MODEL_PROVIDER=openai.")
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"), max_retries=8, timeout=120.0
        )
        self.model = os.getenv("MODEL", "gpt-4.1-nano")
        self.reasoning = self.model.lower().startswith(("gpt-5", "o1", "o3", "o4"))
        self.effort = os.getenv("REASONING_EFFORT", "low")
        if self.reasoning:
            budget = {"low": 4096, "medium": 8192, "high": 16384}
            self.base_body = {
                "model": self.model,
                "max_completion_tokens": budget.get(self.effort, 4096),
                "reasoning_effort": self.effort,
            }
        else:
            self.base_body = {
                "model": self.model,
                "max_tokens": 3072,
                "temperature": args.temperature,
            }

        tag = self.effort if self.reasoning or self.corpus != "rpp" else "openai"
        self.out_dir = PACKAGE_DIR / "prediction" / "LLM_Reasoning" / CORPUS_DIR[self.corpus]
        self.out_dir.mkdir(parents=True, exist_ok=True)
        default_csv = self.out_dir / f"text_predictions_{tag}_temp{args.temperature}.csv"
        self.out_csv = Path(os.getenv("PREDICT_OUT_CSV", str(default_csv)))
        self.meta_file = self.out_dir / f"batch_meta_{tag}_temp{args.temperature}.json"
        self.max_chunk_bytes = int(os.getenv("MAX_CHUNK_BYTES", "180000000"))

    def load_meta(self):
        if not self.meta_file.exists():
            raise SystemExit(f"No meta file found at {self.meta_file}. Run --mode submit first.")
        with self.meta_file.open() as handle:
            return json.load(handle)

    @staticmethod
    def batches_from_meta(meta):
        if "batches" in meta:
            return meta["batches"]
        return [{"batch_id": meta["batch_id"], "file_id": meta.get("file_id"),
                 "index": meta["index"]}]

    def selected_batches(self, meta):
        batches = self.batches_from_meta(meta)
        if not self.args.batch_id:
            return batches
        selected = [b for b in batches if b["batch_id"] == self.args.batch_id]
        if not selected:
            raise SystemExit(f"Batch {self.args.batch_id} is not present in {self.meta_file}")
        return selected

    def existing_done(self):
        if not self.out_csv.exists():
            return set()
        try:
            existing = pd.read_csv(self.out_csv)
        except pd.errors.EmptyDataError:
            return set()
        columns = DEDUP_COLUMNS[self.corpus]
        if "q_hat" not in existing or not set(columns) <= set(existing.columns):
            return set()
        valid = existing[existing["q_hat"].notna()]
        return {tuple(native(value) for value in row) for row in valid[columns].itertuples(index=False, name=None)}

    def submit(self):
        done = self.existing_done()
        if done:
            print(f"Resuming: {len(done)} valid claim/run pairs already done — skipping.")

        index, requests, text_cache = [], [], {}
        for item in tqdm(self.data["items"], desc="Building requests"):
            cache_key = identity_fields(self.corpus, item)[0]
            if self.corpus == "rpp":
                cache_key = item["title"]
            if cache_key not in text_cache:
                text_cache[cache_key] = read_paper(self.corpus, item, self.args.max_chars)
            paper = text_cache[cache_key]
            if paper is None:
                print(f"Skipping (missing OCR text): {item_label(self.corpus, item)}")
                continue
            messages = messages_for(self.corpus, self.data, item, paper)
            for run_id in range(self.runs):
                if resume_key(self.corpus, item, run_id) in done:
                    continue
                custom_id = f"r{len(index):06d}"
                index.append([custom_id, *identity_fields(self.corpus, item), run_id])
                requests.append({
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {**self.base_body, "messages": messages},
                })

        if not requests:
            print("Nothing to submit — all runs already done.")
            return

        chunks = list(chunk_requests(index, requests, self.max_chunk_bytes))
        print(f"Submitting {len(requests)} requests for {self.model}, split into "
              f"{len(chunks)} batch(es)...")
        batches = []
        for number, (chunk_index, chunk) in enumerate(chunks, start=1):
            with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as handle:
                for request in chunk:
                    handle.write(json.dumps(request) + "\n")
                temp_path = handle.name
            size_mb = os.path.getsize(temp_path) / 1024 / 1024
            try:
                with open(temp_path, "rb") as handle:
                    uploaded = self.client.files.create(file=handle, purpose="batch")
            finally:
                os.unlink(temp_path)
            batch = self.client.batches.create(
                input_file_id=uploaded.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={
                    "model": self.model,
                    "corpus": self.corpus,
                    "runs": str(self.runs),
                    "chunk": f"{number}/{len(chunks)}",
                },
            )
            print(f"  [{number}/{len(chunks)}] {len(chunk)} requests, {size_mb:.1f}MB "
                  f"→ batch {batch.id} ({batch.status})")
            batches.append({"batch_id": batch.id, "file_id": uploaded.id, "index": chunk_index})

        with self.meta_file.open("w") as handle:
            json.dump({"model": self.model, "corpus": self.corpus,
                       "runs": self.runs, "batches": batches}, handle, indent=2)
        print(f"Meta saved → {self.meta_file}")
        command = f"python prediction/llm_pred/predict_text_batch.py --corpus {self.corpus}"
        print(f"Check status: {command} --mode status")
        print(f"Collect:      {command} --mode collect")

    def status(self):
        batch_ids = ([self.args.batch_id] if self.args.batch_id else
                     [b["batch_id"] for b in self.batches_from_meta(self.load_meta())])
        totals = [0, 0, 0]
        for batch_id in batch_ids:
            batch = self.client.batches.retrieve(batch_id)
            counts = batch.request_counts
            print(f"Batch:     {batch_id}\nStatus:    {batch.status}\n"
                  f"Requests:  total={counts.total} completed={counts.completed} failed={counts.failed}")
            if batch.output_file_id:
                print(f"Output:    {batch.output_file_id}")
            if batch.error_file_id:
                print(f"Errors:    {batch.error_file_id}")
            if batch.errors:
                for error in batch.errors.data:
                    print(f"  ERROR [{error.code}]: {error.message}")
            totals = [totals[0] + counts.total, totals[1] + counts.completed,
                      totals[2] + counts.failed]
        if len(batch_ids) > 1:
            print(f"TOTAL: total={totals[0]} completed={totals[1]} failed={totals[2]}")

    def item_from_payload(self, payload):
        identity, run_id = tuple(payload[:-1]), int(payload[-1])
        item = self.lookup.get(identity)
        if item is None and self.corpus == "rpp":
            item = next((candidate for key, candidate in self.lookup.items()
                         if key[0] == identity[0]), None)
        if item is None:
            raise KeyError(f"Unknown {self.corpus} identity: {identity}")
        return item, run_id

    def collect(self):
        meta = self.load_meta()
        batches = self.selected_batches(meta)
        pending = []
        for batch_info in batches:
            status = self.client.batches.retrieve(batch_info["batch_id"]).status
            if status != "completed":
                pending.append((batch_info["batch_id"], status))
        if pending:
            for batch_id, status in pending:
                print(f"Batch not complete: {batch_id} ({status})")
            raise SystemExit(1)

        try:
            existing = pd.read_csv(self.out_csv) if self.out_csv.exists() else pd.DataFrame()
            if "q_hat" in existing:
                existing = existing[existing["q_hat"].notna()]
        except pd.errors.EmptyDataError:
            existing = pd.DataFrame()

        records, errors, legacy_offset = [], 0, 0
        for batch_info in batches:
            entries = batch_info["index"]
            has_ids = bool(entries and isinstance(entries[0][0], str)
                           and entries[0][0].startswith("r"))
            if has_ids:
                index = {entry[0]: entry[1:] for entry in entries}
            else:
                index = {f"r{legacy_offset + i:06d}": entry for i, entry in enumerate(entries)}
                legacy_offset += len(entries)

            batch = self.client.batches.retrieve(batch_info["batch_id"])
            print(f"Downloading output for {batch_info['batch_id']} ({batch.output_file_id})...")
            content = self.client.files.content(batch.output_file_id).text
            for line in content.splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                custom_id = obj["custom_id"]
                if custom_id not in index:
                    print(f"Unknown custom_id: {custom_id}")
                    continue
                item, run_id = self.item_from_payload(index[custom_id])
                if obj.get("error"):
                    print(f"Error for {custom_id} ({item_label(self.corpus, item)} run {run_id}): "
                          f"{obj['error']}")
                    errors += 1
                    continue
                text = obj["response"]["body"]["choices"][0]["message"]["content"] or ""
                records.append(response_record(self.corpus, item, run_id, text))

            if batch.error_file_id:
                error_path = self.out_dir / f"batch_errors_{batch_info['batch_id']}.jsonl"
                error_path.write_text(self.client.files.content(batch.error_file_id).text)
                print(f"Error details → {error_path}")

        frames = [frame for frame in (existing, pd.DataFrame(records)) if not frame.empty]
        result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        dedup = DEDUP_COLUMNS[self.corpus]
        if not result.empty:
            result = result.drop_duplicates(subset=dedup, keep="last").reset_index(drop=True)
        result.to_csv(self.out_csv, index=False)
        print(f"Saved {len(result)} rows → {self.out_csv} ({errors} errors)")
        if len(result) and {"ground_truth", "prediction"} <= set(result.columns):
            truth = result["ground_truth"].map({"yes": "replicable", "no": "unreplicable"})
            print(f"Accuracy: {(result['prediction'] == truth).mean():.3f}")

    def display_model(self):
        if self.args.mode in {"status", "collect"} and self.meta_file.exists():
            try:
                return self.load_meta().get("model", self.model)
            except (json.JSONDecodeError, OSError):
                pass
        return self.model

    def run(self):
        print(f"[predict_text_batch:{self.corpus}] model={self.display_model()} "
              f"mode={self.args.mode} out_dir={self.out_dir}")
        {"submit": self.submit, "status": self.status, "collect": self.collect}[self.args.mode]()


def main():
    BatchJob(parse_args()).run()


if __name__ == "__main__":
    main()
