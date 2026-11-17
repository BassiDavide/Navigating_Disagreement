#!/usr/bin/env python3
"""Rehydrate the Navigating_Disagreement datasets with the YouTube Data API v3.

  1. Paste your YouTube Data API v3 key into API_KEY below
     (or set the YOUTUBE_API_KEY environment variable).
  2. Run from the Data/ folder:   python rehydrate.py
     or rehydrate only some datasets:   python rehydrate.py --datasets test,training

Restores comment texts and author names and writes the datasets, in their original format, to Data/rehydrated/.
Downloads are cached, so an interrupted run (e.g. daily quota exhausted) resumes where it stopped.
No third-party packages are required. See Data/README.md for details.
"""

API_KEY = ""  # <-- paste your YouTube Data API v3 key here

import argparse
import io
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

API_URL = "https://www.googleapis.com/youtube/v3/comments"
BATCH_SIZE = 50          # max comment IDs per comments.list request (1 quota unit each)
DAILY_QUOTA = 10_000     # default free quota of a Google Cloud project
MAX_RETRIES = 5
HERE = Path(__file__).resolve().parent
DATASETS = ("test", "training", "active_users", "total")  # processed in this order (smallest first)


class QuotaExceeded(Exception):
    pass


# ----------------------------------------------------------------------------- YouTube API
def api_get(params):
    """GET the comments endpoint, retrying transient errors with exponential backoff."""
    url = API_URL + "?" + urllib.parse.urlencode(params)
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            try:
                err = json.loads(body)["error"]
                reasons = {x.get("reason") for x in err.get("errors", [])}
                message = err.get("message", body)
            except (ValueError, KeyError, TypeError):
                reasons, message = set(), body
            if reasons & {"quotaExceeded", "dailyLimitExceeded"}:
                raise QuotaExceeded(message)
            transient = e.code >= 500 or e.code == 429 or "rateLimitExceeded" in reasons
            if not transient:  # e.g. invalid key, API not enabled
                sys.exit(f"\nYouTube API error {e.code} ({', '.join(sorted(r for r in reasons if r)) or 'unknown'}): "
                         f"{message}\nProgress so far is saved in the cache; re-run to resume.")
            error = f"HTTP {e.code}: {message}"
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            error = str(e)
        wait = 2 ** attempt
        print(f"  request failed ({error}); retrying in {wait}s", file=sys.stderr)
        time.sleep(wait)
    sys.exit("\nToo many failed requests. Progress is saved in the cache; re-run to resume.")


class CommentCache:
    """Append-only JSONL cache: one line per comment ID ({"id", "text", "author"}; text/author null if unavailable)."""

    def __init__(self, path):
        self.path = path
        self.data = {}
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except ValueError:  # partially written last line after a crash
                        continue
                    self.data[r["id"]] = r if r.get("text") is not None else None

    def get(self, cid):
        return self.data.get(cid)

    def fetch(self, ids, api_key, label):
        todo = sorted(i for i in ids if i and i not in self.data)
        if not todo:
            print(f"[{label}] all {len(ids)} comments already cached")
            return
        n_batches = -(-len(todo) // BATCH_SIZE)
        days = n_batches / DAILY_QUOTA
        print(f"[{label}] fetching {len(todo)} comments in {n_batches} requests"
              + (f" (~{days:.1f} days of default quota; re-run each day to continue)" if days > 0.9 else ""))
        with open(self.path, "a", encoding="utf-8") as f:
            for b in range(n_batches):
                batch = todo[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
                data = api_get({"part": "snippet", "id": ",".join(batch), "textFormat": "plainText", "key": api_key})
                found = {}
                for item in data.get("items", []):
                    s = item.get("snippet", {})
                    found[item["id"]] = {"id": item["id"], "text": s.get("textOriginal", s.get("textDisplay")),
                                         "author": s.get("authorDisplayName")}
                for cid in batch:
                    rec = found.get(cid, {"id": cid, "text": None, "author": None})
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    self.data[cid] = rec if rec["text"] is not None else None
                f.flush()
                if (b + 1) % 20 == 0 or b + 1 == n_batches:
                    print(f"  [{b + 1}/{n_batches}] requests done")


def text(cache, cid):
    c = cache.get(cid)
    return c["text"] if c else None


def author(cache, cid):
    c = cache.get(cid)
    return c["author"] if c else None


# ----------------------------------------------------------------------------- per-dataset logic
def read_jsonl(lines):
    return [json.loads(l) for l in lines if l.strip()]


def write_jsonl(path, rows, compact=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    sep = (",", ":") if compact else None
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False, separators=sep) + "\n" for r in rows)


class Test:
    name = "test"

    def __init__(self, data_dir):
        with open(data_dir / "Test_Data.jsonl", encoding="utf-8") as f:
            self.rows = read_jsonl(f)

    def ids(self):
        return {r["CommentID"] for r in self.rows} | {r["ResponseToID"] for r in self.rows if r["ResponseToID"]}

    def write(self, cache, out_dir):
        out = [{"CommentID": r["CommentID"],
                "ParentCommentText": text(cache, r["ResponseToID"]) if r["ResponseToID"] else r["ParentCommentText"],
                "CommentText": text(cache, r["CommentID"]),
                "AuthorName": author(cache, r["CommentID"]),
                "Stance_Label": r["Stance_Label"]} for r in self.rows]
        write_jsonl(out_dir / "Test_Data.jsonl", out)
        return len(out), sum(r["CommentText"] is None or r["ParentCommentText"] is None for r in out)


class Training:
    name = "training"
    META = ("hypothesis", "label", "original_stance", "input_type")

    def __init__(self, data_dir):
        with zipfile.ZipFile(data_dir / "Training_Data.jsonl.zip") as z:
            self.rows = read_jsonl(io.TextIOWrapper(z.open("Training_Data.jsonl"), encoding="utf-8"))

    def ids(self):
        return {r["CommentID"] for r in self.rows} | {r["ResponseToID"] for r in self.rows if r.get("ResponseToID")}

    def write(self, cache, out_dir):
        out = []
        for r in self.rows:
            comment = text(cache, r["CommentID"])
            if r["input_type"] == "child_only":
                premise = None if comment is None else f"Context: {comment}"
            else:
                ctx = text(cache, r["ResponseToID"]) if r.get("ResponseToID") else r["Context"]
                premise = None if comment is None or ctx is None else f"Context: {ctx}\nComment: {comment}"
            out.append({"premise": premise, **{k: r[k] for k in self.META}})
        write_jsonl(out_dir / "Training_Data.jsonl", out)
        return len(out), sum(r["premise"] is None for r in out)


class ActiveUsers:
    name = "active_users"

    def __init__(self, data_dir):
        with zipfile.ZipFile(data_dir / "active_users.json.zip") as z:
            self.users = json.loads(z.read("active_users.json"))

    def ids(self):
        ids = set()
        for u in self.users.values():
            for c in u["comments"]:
                ids.add(c["commentID"])
                if c["Level"] > 0:
                    ids.add(c["parent"])
        return ids

    def write(self, cache, out_dir):
        out, n, missing = {}, 0, 0
        for key, u in self.users.items():
            comments = []
            for c in u["comments"]:
                d = {}
                for k, v in c.items():
                    d[k] = v
                    if k == "commentID":
                        d["comment_text"] = text(cache, v)
                    elif k == "parent" and c["Level"] > 0:
                        d["context"] = text(cache, v)  # the parent comment; for Level 0 the context is the video title
                n += 1
                missing += d["comment_text"] is None or d.get("context") is None
                comments.append(d)
            # key the user by their YouTube handle, as in the original data (fallback: the placeholder key)
            handle = next((author(cache, c["commentID"]) for c in u["comments"] if author(cache, c["commentID"])), None)
            new_key = handle if handle and handle not in out else key
            out[new_key] = {k: (comments if k == "comments" else v) for k, v in u.items()}
        path = out_dir / "active_users.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        return n, missing


class Total:
    name = "total"

    def __init__(self, data_dir):
        self.zips = sorted((data_dir / "Dataset_Total").glob("*.zip"))

    def _entries(self):
        for zp in self.zips:
            with zipfile.ZipFile(zp) as z:
                for name in z.namelist():
                    if name.endswith(".jsonl"):
                        yield name, read_jsonl(io.TextIOWrapper(z.open(name), encoding="utf-8"))

    def ids(self):
        return {r["CommentID"] for _, rows in self._entries() for r in rows}

    def write(self, cache, out_dir):
        n = missing = 0
        for name, rows in self._entries():
            out = []
            for r in rows:
                d = {}
                for k, v in r.items():
                    d[k] = v
                    if k == "VideoID":
                        d["CommentText"] = text(cache, r["CommentID"])
                        d["AuthorName"] = author(cache, r["CommentID"])
                out.append(d)
                n += 1
                missing += d["CommentText"] is None
            write_jsonl(out_dir / "Dataset_Total" / name, out, compact=True)
        return n, missing


LOADERS = {"test": Test, "training": Training, "active_users": ActiveUsers, "total": Total}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default=",".join(DATASETS),
                    help=f"comma-separated subset of {', '.join(DATASETS)} (default: all)")
    ap.add_argument("--data-dir", type=Path, default=HERE, help="folder with the dehydrated files (default: Data/)")
    ap.add_argument("--out-dir", type=Path, default=HERE / "rehydrated", help="output folder (default: Data/rehydrated/)")
    args = ap.parse_args()

    selected = [d.strip() for d in args.datasets.split(",") if d.strip()]
    unknown = set(selected) - set(DATASETS)
    if unknown:
        sys.exit(f"Unknown dataset(s): {', '.join(sorted(unknown))}. Choose from: {', '.join(DATASETS)}")
    api_key = (API_KEY or os.environ.get("YOUTUBE_API_KEY", "")).strip()
    if not api_key:
        sys.exit("No API key found. Paste your YouTube Data API v3 key into API_KEY at the top of rehydrate.py, "
                 "or set the YOUTUBE_API_KEY environment variable. See Data/README.md.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache = CommentCache(args.out_dir / "comments_cache.jsonl")
    all_ids, done = set(), []
    try:
        for name in DATASETS:
            if name not in selected:
                continue
            ds = LOADERS[name](args.data_dir)
            ids = ds.ids()
            all_ids |= ids
            cache.fetch(ids, api_key, name)
            n, missing = ds.write(cache, args.out_dir)
            done.append(name)
            print(f"[{name}] written: {n} records, {missing} with unavailable text\n")
    except QuotaExceeded as e:
        print(f"\nDaily API quota exhausted ({e}).\nCompleted datasets: {', '.join(done) or 'none'}. "
              f"Progress is cached in {cache.path}: run the script again after the quota resets "
              f"(midnight Pacific Time) to continue.")
        sys.exit(2)

    missing = sorted(i for i in all_ids if cache.get(i) is None)
    with open(args.out_dir / "missing_comments.txt", "w", encoding="utf-8") as f:
        f.writelines(i + "\n" for i in missing)
    print(f"Done. {len(all_ids) - len(missing)}/{len(all_ids)} comments retrieved; {len(missing)} unavailable "
          f"(deleted, hidden or private) listed in {args.out_dir / 'missing_comments.txt'}")


if __name__ == "__main__":
    main()
