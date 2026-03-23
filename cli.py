from __future__ import annotations

import argparse
from pathlib import Path

import requests

BASE_URL = "http://127.0.0.1:8000"


def cmd_upload(args: argparse.Namespace) -> None:
    path = Path(args.path)
    with path.open("rb") as f:
        r = requests.post(f"{BASE_URL}/books/upload", files={"file": (path.name, f, "text/plain")}, timeout=60)
    print(r.status_code, r.json())


def cmd_find(args: argparse.Namespace) -> None:
    r = requests.post(f"{BASE_URL}/search/snippets", json={"query": args.query, "top_k": args.top_k}, timeout=60)
    data = r.json()
    print("Query:", data.get("query"))
    for i, s in enumerate(data.get("snippets", []), start=1):
        print(f"[{i}] {s['book']} score={s['score']} {s['offset_start']}-{s['offset_end']}")
        print(s["quote"])
        print("-")
    if not data.get("snippets"):
        print("Nothing found")


def cmd_ask(args: argparse.Namespace) -> None:
    r = requests.post(f"{BASE_URL}/ask", json={"question": args.question, "top_k": args.top_k}, timeout=60)
    data = r.json()
    print("Answer:", data.get("answer"))
    for i, s in enumerate(data.get("sources", []), start=1):
        print(f"[{i}] {s['book']} score={s['score']} {s['offset_start']}-{s['offset_end']}")
        print(s["quote"])
        print("-")


def main() -> None:
    parser = argparse.ArgumentParser(description="CLI client for Smart Book Search")
    sub = parser.add_subparsers(required=True)

    up = sub.add_parser("upload")
    up.add_argument("path")
    up.set_defaults(func=cmd_upload)

    find = sub.add_parser("find")
    find.add_argument("query")
    find.add_argument("--top-k", type=int, default=5)
    find.set_defaults(func=cmd_find)

    ask = sub.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=5)
    ask.set_defaults(func=cmd_ask)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
