"""Create text embeddings for RPP, CB, or SSRP with one corpus-aware CLI.

RPP and SSRP share the one-study-per-paper pipeline. CB retains its
paper-level chunk sharing and effect-level context vectors. Synchronous mode
works for all corpora; asynchronous Batch API modes work for the single-study
RPP/SSRP layout.

Examples::

    rep_env/bin/python prediction/embed_pred/embed_text_batch.py --corpus rpp --mode sync
    rep_env/bin/python prediction/embed_pred/embed_text_batch.py --corpus cb --limit 5
    rep_env/bin/python prediction/embed_pred/embed_text_batch.py --corpus ssrp --mode submit
"""

import argparse
import hashlib
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
from rpp_outcomes import canonical_rpp_outcomes_for_titles  # noqa: E402

try:
    import tiktoken
except ImportError:
    tiktoken = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=["rpp", "cb", "ssrp"], default="rpp")
    parser.add_argument("--mode", choices=["sync", "submit", "status", "collect"], default="sync")
    parser.add_argument("--limit", type=int, default=0, help="process only the first N rows")
    parser.add_argument("--max-chars", type=int, default=0, help="truncate paper text")
    parser.add_argument("--chunk-tokens", type=int, default=1000)
    parser.add_argument("--overlap-tokens", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128, help="inputs per sync request")
    parser.add_argument("--batch-id", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--context", choices=["none", "query", "prefix", "both"], default="query",
        help="embed the locator separately, prefix it to chunks, do both, or omit it",
    )
    return parser.parse_args()


def slugify(value):
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(value)).strip("_").lower()[:80]
    suffix = hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:8]
    return f"{slug}_{suffix}"


def l2(values):
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        return array / max(float(np.linalg.norm(array)), 1e-12)
    return array / np.clip(np.linalg.norm(array, axis=1, keepdims=True), 1e-12, None)


class Chunker:
    CHARS_PER_TOKEN = 4

    def __init__(self, chunk_tokens, overlap_tokens):
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens
        if overlap_tokens >= chunk_tokens:
            raise SystemExit("--overlap-tokens must be smaller than --chunk-tokens")
        self.tokenizer = None
        if tiktoken is not None:
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
            except Exception as error:
                print(f"Could not load tiktoken encoding ({error}); using character-based chunking.")
        if self.tokenizer is None:
            print("Using character-based chunking (~4 chars/token).")

    def split(self, text):
        stride = self.chunk_tokens - self.overlap_tokens
        chunks = []
        if self.tokenizer is not None:
            tokens = self.tokenizer.encode(text)
            cursor = 0
            for start_token in range(0, max(len(tokens), 1), stride):
                window = tokens[start_token:start_token + self.chunk_tokens]
                if not window:
                    break
                body = self.tokenizer.decode(window)
                n_tokens = len(window)
                start = text.find(body, cursor)
                if start == -1:
                    body = body.strip("�")
                    start = text.find(body, cursor)
                    n_tokens = len(self.tokenizer.encode(body))
                if start == -1:
                    print(f"  warning: could not locate chunk at token {start_token}; offsets approximate")
                    start = cursor
                cursor = start + 1
                if body.strip():
                    chunks.append({"text": body, "n_tokens": n_tokens,
                                   "start": start, "end": start + len(body)})
                if start_token + self.chunk_tokens >= len(tokens):
                    break
            return chunks

        size = self.chunk_tokens * self.CHARS_PER_TOKEN
        char_stride = stride * self.CHARS_PER_TOKEN
        for start in range(0, max(len(text), 1), char_stride):
            body = text[start:start + size]
            if body.strip():
                chunks.append({"text": body, "n_tokens": len(body) // self.CHARS_PER_TOKEN,
                               "start": start, "end": start + len(body)})
            if start + size >= len(text):
                break
        return chunks


class EmbeddingJob:
    def __init__(self, args):
        self.args = args
        self.want_query = args.context in {"query", "both"}
        self.want_prefix = args.context in {"prefix", "both"}
        self.chunker = Chunker(args.chunk_tokens, args.overlap_tokens)
        load_dotenv(REPO_ROOT / ".env")
        if os.getenv("MODEL_PROVIDER", "openai") != "openai":
            raise SystemExit("Embeddings only support MODEL_PROVIDER=openai.")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("EMBED_MODEL", "text-embedding-3-large")

    def embed(self, texts):
        vectors = []
        for start in range(0, len(texts), self.args.batch_size):
            response = self.client.embeddings.create(
                model=self.model, input=texts[start:start + self.args.batch_size]
            )
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda item: item.index))
        return vectors

    def make_dirs(self, *directories):
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


class SingleStudyJob(EmbeddingJob):
    def __init__(self, args):
        super().__init__(args)
        self.corpus = args.corpus
        if self.corpus == "rpp":
            self.frame = pd.read_csv(PACKAGE_DIR / "data" / "rpp_data_cleaned.csv", encoding="latin1")
            self.out_dir = PACKAGE_DIR / f"embeddings_{self.model}_{args.context}"
        else:
            self.frame = pd.read_csv(PACKAGE_DIR / "data" / "ssrp_data_cleaned.csv")
            self.out_dir = PACKAGE_DIR / "prediction" / "embed_pred" / "SSRP" / f"embeddings_{self.model}_{args.context}"
        if args.limit:
            self.frame = self.frame.head(args.limit)
        self.emb_dir = self.out_dir / "emb"
        self.ctx_dir = self.out_dir / "ctx"
        self.chunk_dir = self.out_dir / "chunks"
        self.index_csv = self.out_dir / "index.csv"
        self.pooled_npy = self.out_dir / "pooled.npy"
        self.context_npy = self.out_dir / "context.npy"
        self.meta_file = self.out_dir / "batch_meta.json"
        self.make_dirs(self.out_dir, self.emb_dir, self.ctx_dir, self.chunk_dir)

    def source_rows(self):
        if self.corpus == "rpp":
            for _, row in self.frame.iterrows():
                yield row, row["Study Title (O)"], row["Replicated study number (R)"]
        else:
            for _, row in self.frame.iterrows():
                yield row, row["Study Title (O)"], int(row["Study Num"])

    def read_text(self, identifier, title):
        if self.corpus == "rpp":
            return prompt_text.read_paper_text(title, self.args.max_chars)
        return prompt_text_ssrp.read_paper_text(identifier, self.args.max_chars)

    def locator(self, identifier, title):
        if self.corpus == "rpp":
            return "\n".join(prompt_text.build_locator(self.frame, title, identifier))
        return "\n".join(prompt_text_ssrp.build_locator(self.frame, identifier))

    def truth(self, row, title):
        if self.corpus == "rpp":
            outcome = canonical_rpp_outcomes_for_titles([title]).iloc[0]
            if pd.isna(outcome):
                raise ValueError(f"Canonical RPP outcome missing for {title!r}")
            return "yes" if int(outcome) == 1 else "no"
        return str(row["Replicate (R)"]).strip().lower()

    def build_plan(self):
        plan, skipped = [], []
        for row, title, identifier in tqdm(list(self.source_rows()), desc="Chunking papers"):
            slug = slugify(title)
            if not self.args.overwrite and (self.emb_dir / f"{slug}.npy").exists():
                skipped.append(title)
                continue
            text = self.read_text(identifier, title)
            if text is None:
                print(f"Skipping (missing OCR text): {title}")
                continue
            chunks = self.chunker.split(text)
            if not chunks:
                print(f"Skipping (empty text): {title}")
                continue
            context = self.locator(identifier, title)
            for chunk in chunks:
                chunk["input"] = f"{context}\n\n{chunk['text']}" if self.want_prefix else chunk["text"]
            id_key = "study_number" if self.corpus == "rpp" else "study_num"
            identifier = identifier.item() if isinstance(identifier, np.generic) else identifier
            plan.append({
                "title": title,
                "slug": slug,
                id_key: int(identifier) if self.corpus == "ssrp" else identifier,
                "ground_truth": self.truth(row, title),
                "n_chars": len(text),
                "context": context,
                "chunks": chunks,
            })
        return plan, skipped

    def save_matrix(self, entry, vectors, context_vector=None):
        np.save(self.emb_dir / f"{entry['slug']}.npy", l2(vectors))
        if context_vector is not None:
            np.save(self.ctx_dir / f"{entry['slug']}.npy", l2(context_vector))
        chunks = [
            {"i": index, "n_tokens": chunk["n_tokens"], "start": chunk["start"],
             "end": chunk["end"], "preview": chunk.get("text", "")[:120].replace("\n", " ")}
            for index, chunk in enumerate(entry["chunks"])
        ]
        metadata = {key: value for key, value in entry.items() if key not in {"chunks", "n_chars"}}
        metadata.update({
            "n_chunks": len(chunks), "chunk_tokens": self.args.chunk_tokens,
            "overlap_tokens": self.args.overlap_tokens, "model": self.model,
            "context_mode": self.args.context, "chunks": chunks,
        })
        with (self.chunk_dir / f"{entry['slug']}.json").open("w") as handle:
            json.dump(metadata, handle, indent=2)

    def rebuild_index(self):
        rows, pooled, contexts = [], [], []
        for matrix_path in sorted(self.emb_dir.glob("*.npy")):
            slug = matrix_path.stem
            matrix = np.load(matrix_path)
            with (self.chunk_dir / f"{slug}.json").open() as handle:
                metadata = json.load(handle)
            title = metadata["title"]
            context_path = self.ctx_dir / f"{slug}.npy"
            has_context = context_path.exists()
            pooled.append(l2(matrix.mean(axis=0)))
            if has_context:
                contexts.append(np.load(context_path))

            if self.corpus == "rpp":
                match = self.frame[self.frame["Study Title (O)"] == title]
                identifier = metadata.get(
                    "study_number",
                    match["Replicated study number (R)"].iloc[0] if len(match) else "",
                )
                outcome = canonical_rpp_outcomes_for_titles([title]).iloc[0]
                ground_truth = ("yes" if int(outcome) == 1 else "no") if pd.notna(outcome) else ""
                id_fields = {"study_number": identifier}
            else:
                match = self.frame[self.frame["Study Title (O)"] == title]
                identifier = metadata.get("study_num", int(match["Study Num"].iloc[0]) if len(match) else "")
                ground_truth = metadata.get(
                    "ground_truth",
                    str(match["Replicate (R)"].iloc[0]).strip().lower() if len(match) else "",
                )
                id_fields = {"study_num": identifier}
            rows.append({
                "slug": slug, "title": title, **id_fields,
                "n_chunks": matrix.shape[0], "dim": matrix.shape[1],
                "has_context": has_context, "ground_truth": ground_truth,
            })

        if not rows:
            print("No embeddings on disk yet — nothing to index.")
            return
        pd.DataFrame(rows).to_csv(self.index_csv, index=False)
        pooled_matrix = np.asarray(pooled, dtype=np.float32)
        np.save(self.pooled_npy, pooled_matrix)
        unit = "files" if self.corpus == "rpp" else "studies"
        print(f"Index: {len(rows)} {unit} → {self.index_csv}")
        print(f"Pooled: {pooled_matrix.shape} → {self.pooled_npy}")
        if len(contexts) == len(rows):
            context_matrix = np.asarray(contexts, dtype=np.float32)
            np.save(self.context_npy, context_matrix)
            print(f"Context: {context_matrix.shape} → {self.context_npy}")
        elif contexts:
            print(f"Context: only {len(contexts)}/{len(rows)} rows available; not writing matrix.")

    def sync(self):
        plan, skipped = self.build_plan()
        if skipped:
            print(f"Resuming: {len(skipped)} files already embedded — skipping.")
        for entry in tqdm(plan, desc="Embedding files"):
            texts = [chunk["input"] for chunk in entry["chunks"]]
            if self.want_query:
                texts.append(entry["context"])
            vectors = self.embed(texts)
            context_vector = vectors.pop() if self.want_query else None
            self.save_matrix(entry, vectors, context_vector)
        self.rebuild_index()

    def load_meta(self):
        if not self.meta_file.exists():
            raise SystemExit(f"No meta file found at {self.meta_file}. Run --mode submit first.")
        with self.meta_file.open() as handle:
            return json.load(handle)

    def submit(self):
        plan, skipped = self.build_plan()
        if skipped:
            print(f"Resuming: {len(skipped)} files already embedded — skipping.")
        if not plan:
            print("Nothing to submit.")
            return
        index, requests = [], []
        for entry in plan:
            for chunk_index, chunk in enumerate(entry["chunks"]):
                index.append([entry["slug"], chunk_index])
                requests.append({
                    "custom_id": f"e{len(index) - 1:06d}", "method": "POST",
                    "url": "/v1/embeddings", "body": {"model": self.model, "input": chunk["input"]},
                })
            if self.want_query:
                index.append([entry["slug"], -1])
                requests.append({
                    "custom_id": f"e{len(index) - 1:06d}", "method": "POST",
                    "url": "/v1/embeddings", "body": {"model": self.model, "input": entry["context"]},
                })
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as handle:
            for request in requests:
                handle.write(json.dumps(request) + "\n")
            temp_path = handle.name
        try:
            with open(temp_path, "rb") as handle:
                uploaded = self.client.files.create(file=handle, purpose="batch")
        finally:
            os.unlink(temp_path)
        batch = self.client.batches.create(
            input_file_id=uploaded.id, endpoint="/v1/embeddings", completion_window="24h",
            metadata={"model": self.model, "corpus": self.corpus, "n_files": str(len(plan))},
        )
        compact_plan = [
            {key: value for key, value in entry.items() if key != "chunks"} |
            {"chunks": [{key: value for key, value in chunk.items() if key not in {"text", "input"}}
                        for chunk in entry["chunks"]]}
            for entry in plan
        ]
        with self.meta_file.open("w") as handle:
            json.dump({
                "batch_id": batch.id, "file_id": uploaded.id, "model": self.model,
                "corpus": self.corpus, "chunk_tokens": self.args.chunk_tokens,
                "overlap_tokens": self.args.overlap_tokens, "context_mode": self.args.context,
                "index": index, "plan": compact_plan,
            }, handle, indent=2)
        print(f"Batch submitted: {batch.id} ({batch.status})\nMeta saved → {self.meta_file}")

    def status(self):
        meta = self.load_meta()
        batch_id = self.args.batch_id or meta["batch_id"]
        batch = self.client.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(f"Batch: {batch_id}\nStatus: {batch.status}\n"
              f"Requests: total={counts.total} completed={counts.completed} failed={counts.failed}")

    def collect(self):
        meta = self.load_meta()
        batch_id = self.args.batch_id or meta["batch_id"]
        batch = self.client.batches.retrieve(batch_id)
        if batch.status != "completed":
            raise SystemExit(f"Batch not complete yet — status: {batch.status}")
        index = {f"e{i:06d}": tuple(value) for i, value in enumerate(meta["index"])}
        plan = {entry["slug"]: entry for entry in meta["plan"]}
        content = self.client.files.content(batch.output_file_id).text
        by_file, errors = {}, 0
        for line in content.splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            custom_id = obj["custom_id"]
            if custom_id not in index:
                print(f"Unknown custom_id: {custom_id}")
                continue
            slug, chunk_index = index[custom_id]
            if obj.get("error"):
                print(f"Error for {custom_id} ({slug} chunk {chunk_index}): {obj['error']}")
                errors += 1
                continue
            by_file.setdefault(slug, {})[chunk_index] = obj["response"]["body"]["data"][0]["embedding"]
        want_context = meta.get("context_mode", "none") in {"query", "both"}
        written = 0
        for slug, vectors_by_index in by_file.items():
            entry = plan[slug]
            expected = len(entry["chunks"])
            missing = sorted(set(range(expected)) - set(vectors_by_index))
            if missing or (want_context and -1 not in vectors_by_index):
                print(f"Incomplete — {slug}; skipping file.")
                continue
            self.save_matrix(entry, [vectors_by_index[i] for i in range(expected)],
                             vectors_by_index.get(-1))
            written += 1
        print(f"Wrote {written} file matrices ({errors} chunk errors)")
        self.rebuild_index()
        if batch.error_file_id:
            (self.out_dir / "batch_errors.jsonl").write_text(
                self.client.files.content(batch.error_file_id).text
            )

    def run(self):
        print(f"[embed_text_batch:{self.corpus}] model={self.model} mode={self.args.mode} "
              f"context={self.args.context} out_dir={self.out_dir}")
        {"sync": self.sync, "submit": self.submit,
         "status": self.status, "collect": self.collect}[self.args.mode]()


class CBEmbeddingJob(EmbeddingJob):
    def __init__(self, args):
        super().__init__(args)
        if args.mode != "sync":
            raise SystemExit("CB currently supports --mode sync only.")
        self.frame = pd.read_csv(PACKAGE_DIR / "data" / "cb_data_cleaned.csv")
        if args.limit:
            self.frame = self.frame.head(args.limit)
        self.out_dir = PACKAGE_DIR / "prediction" / "embed_pred" / "CB" / f"embeddings_{self.model}_{args.context}"
        self.emb_dir = self.out_dir / "emb"
        self.ctx_dir = self.out_dir / "ctx"
        self.chunk_dir = self.out_dir / "chunks"
        self.effect_dir = self.out_dir / "effects"
        self.index_csv = self.out_dir / "index.csv"
        self.pooled_npy = self.out_dir / "pooled.npy"
        self.context_npy = self.out_dir / "context.npy"
        self.make_dirs(self.out_dir, self.emb_dir, self.ctx_dir, self.chunk_dir, self.effect_dir)

    def build_plan(self):
        plan = []
        for title, group in self.frame.groupby("Study Title (O)", sort=False):
            first = group.iloc[0]
            text = prompt_text_cb.read_paper_text(first["dir"], self.args.max_chars)
            if text is None:
                print(f"Skipping (missing OCR text): {title}")
                continue
            chunks = self.chunker.split(text)
            if not chunks:
                continue
            effects = []
            for _, row in group.iterrows():
                context = "\n".join(prompt_text_cb.build_locator(
                    title, row["Paper #"], row["Experiment #"], row["Effect #"],
                    row["Description of effect (O)"],
                ))
                effects.append({
                    "effect_slug": slugify(
                        f"{title} exp{int(row['Experiment #'])} eff{int(row['Effect #'])}"
                    ),
                    "paper_num": int(row["Paper #"]),
                    "experiment_num": int(row["Experiment #"]),
                    "effect_num": int(row["Effect #"]),
                    "ground_truth": str(row["Replicate (R)"]).strip().lower(),
                    "context": context,
                })
            plan.append({"title": title, "paper_slug": slugify(title),
                         "base_chunks": chunks, "effects": effects})
        return plan

    def save_chunks(self, entry, vectors, slug=None):
        slug = slug or entry["paper_slug"]
        np.save(self.emb_dir / f"{slug}.npy", l2(vectors))
        chunks = [
            {"i": i, "n_tokens": chunk["n_tokens"], "start": chunk["start"],
             "end": chunk["end"], "preview": chunk["text"][:120].replace("\n", " ")}
            for i, chunk in enumerate(entry["base_chunks"])
        ]
        with (self.chunk_dir / f"{slug}.json").open("w") as handle:
            json.dump({"title": entry["title"], "n_chunks": len(chunks),
                       "chunk_tokens": self.args.chunk_tokens,
                       "overlap_tokens": self.args.overlap_tokens,
                       "model": self.model, "chunks": chunks}, handle, indent=2)

    def save_effect(self, entry, effect, paper_slug, context_vector=None):
        if context_vector is not None:
            np.save(self.ctx_dir / f"{effect['effect_slug']}.npy", l2(context_vector))
        metadata = {key: value for key, value in effect.items() if key != "effect_slug"}
        metadata.update({"title": entry["title"], "paper_slug": paper_slug})
        with (self.effect_dir / f"{effect['effect_slug']}.json").open("w") as handle:
            json.dump(metadata, handle, indent=2)

    def sync(self):
        plan = self.build_plan()
        for entry in tqdm(plan, desc="Embedding papers"):
            if self.want_prefix:
                for effect in entry["effects"]:
                    effect_slug = effect["effect_slug"]
                    embedding_exists = (self.emb_dir / f"{effect_slug}.npy").exists()
                    context_exists = (self.ctx_dir / f"{effect_slug}.npy").exists()
                    complete = embedding_exists and (not self.want_query or context_exists)
                    if not self.args.overwrite and complete:
                        continue
                    texts = [f"{effect['context']}\n\n{chunk['text']}"
                             for chunk in entry["base_chunks"]]
                    if self.want_query:
                        texts.append(effect["context"])
                    vectors = self.embed(texts)
                    context_vector = vectors.pop() if self.want_query else None
                    self.save_chunks(entry, vectors, effect_slug)
                    self.save_effect(entry, effect, effect_slug, context_vector)
                continue

            paper_slug = entry["paper_slug"]
            if self.args.overwrite or not (self.emb_dir / f"{paper_slug}.npy").exists():
                self.save_chunks(entry, self.embed([chunk["text"] for chunk in entry["base_chunks"]]))
            for effect in entry["effects"]:
                context_path = self.ctx_dir / f"{effect['effect_slug']}.npy"
                if self.want_query and (self.args.overwrite or not context_path.exists()):
                    self.save_effect(entry, effect, paper_slug, self.embed([effect["context"]])[0])
                else:
                    self.save_effect(entry, effect, paper_slug)
        self.rebuild_index()

    def rebuild_index(self):
        rows, pooled, contexts, cache = [], [], [], {}
        for metadata_path in sorted(self.effect_dir.glob("*.json")):
            with metadata_path.open() as handle:
                metadata = json.load(handle)
            effect_slug = metadata_path.stem
            paper_slug = metadata["paper_slug"]
            matrix_slug = effect_slug if self.want_prefix else paper_slug
            matrix_path = self.emb_dir / f"{matrix_slug}.npy"
            if not matrix_path.exists():
                continue
            if matrix_slug not in cache:
                cache[matrix_slug] = l2(np.load(matrix_path).mean(axis=0))
            context_path = self.ctx_dir / f"{effect_slug}.npy"
            has_context = context_path.exists()
            rows.append({
                "slug": effect_slug, "paper_slug": paper_slug, "title": metadata["title"],
                "paper_num": metadata["paper_num"], "experiment_num": metadata["experiment_num"],
                "effect_num": metadata["effect_num"], "has_context": has_context,
                "ground_truth": metadata["ground_truth"],
            })
            pooled.append(cache[matrix_slug])
            if has_context:
                contexts.append(np.load(context_path))
        if not rows:
            print("No effects on disk yet — nothing to index.")
            return
        pd.DataFrame(rows).to_csv(self.index_csv, index=False)
        pooled_matrix = np.asarray(pooled, dtype=np.float32)
        np.save(self.pooled_npy, pooled_matrix)
        print(f"Index: {len(rows)} effects across {len(cache)} matrices → {self.index_csv}")
        print(f"Pooled: {pooled_matrix.shape} → {self.pooled_npy}")
        if len(contexts) == len(rows):
            context_matrix = np.asarray(contexts, dtype=np.float32)
            np.save(self.context_npy, context_matrix)
            print(f"Context: {context_matrix.shape} → {self.context_npy}")
        elif contexts:
            print(f"Context: only {len(contexts)}/{len(rows)} effects available; not writing matrix.")

    def run(self):
        print(f"[embed_text_batch:cb] model={self.model} context={self.args.context} "
              f"out_dir={self.out_dir}")
        self.sync()


def main():
    args = parse_args()
    job = CBEmbeddingJob(args) if args.corpus == "cb" else SingleStudyJob(args)
    job.run()


if __name__ == "__main__":
    main()
