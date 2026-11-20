"""One-time offline corpus prep: derive ``{pageid}-dated.md`` from cached HTML."""

from __future__ import annotations

import argparse
import sys
import time

from .corpus import html_path, materialize_md, md_path, require_wikicache


def evidence_pageids(split: str) -> dict[int, str]:
    """``{pageid: title}`` for every article referenced by ``split``."""
    import fanoutqa

    out: dict[int, str] = {}
    if split != "dev":
        raise ValueError(f"split must be 'dev', got {split!r}")
    for q in fanoutqa.load_dev():
        for e in q.necessary_evidence:
            out[e.pageid] = e.title
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=["dev"], default="dev")
    p.add_argument("--check", action="store_true", help="report coverage, convert nothing")
    p.add_argument(
        "--fill-network",
        action="store_true",
        help=(
            "for pages absent from the tarball entirely, fetch the pinned revision from "
            "the live Wikipedia API via the canonical fanoutqa.wiki.wiki_content "
            "(measured: 2 of 4122 evidence pages needed this - 'Pope Francis' and "
            "'Steve Chen'). Requires network; run once."
        ),
    )
    args = p.parse_args(argv)

    cache = require_wikicache()
    print(f"wikicache: {cache}", file=sys.stderr)

    splits = [args.split]
    wanted: dict[int, str] = {}
    for s in splits:
        ids = evidence_pageids(s)
        print(f"  {s}: {len(ids)} unique evidence pageids", file=sys.stderr)
        wanted.update(ids)

    have_md = [pid for pid in wanted if md_path(cache, pid).exists()]
    need = [pid for pid in wanted if pid not in set(have_md)]
    have_html = [pid for pid in need if html_path(cache, pid).exists()]
    missing = [pid for pid in need if not html_path(cache, pid).exists()]
    print(
        f"  total {len(wanted)} pages: {len(have_md)} .md, {len(have_html)} convertible "
        f"from .html, {len(missing)} absent from cache",
        file=sys.stderr,
    )
    if missing:
        print("  absent pageids (rows needing them are dropped):", file=sys.stderr)
        for pid in sorted(missing)[:20]:
            print(f"    {pid}  {wanted[pid]}", file=sys.stderr)

    if missing and args.fill_network:
        import fanoutqa
        from fanoutqa.wiki import wiki_content

        by_pageid: dict[int, object] = {}
        for s in splits:
            qs = fanoutqa.load_dev()
            for q in qs:
                for e in q.necessary_evidence:
                    by_pageid.setdefault(e.pageid, e)
        for pid in list(missing):
            try:
                wiki_content(by_pageid[pid])
                print(f"    fetched {pid} {wanted[pid]}", file=sys.stderr)
                missing.remove(pid)
            except Exception as e:  # noqa: BLE001
                print(f"    FAILED {pid} {wanted[pid]}: {type(e).__name__}", file=sys.stderr)

    if args.check:
        return 0

    t0 = time.time()
    done = failed = 0
    for i, pid in enumerate(have_html, 1):
        if materialize_md(cache, pid) is None:
            failed += 1
        else:
            done += 1
        if i % 200 == 0:
            print(f"    {i}/{len(have_html)} ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    print(f"  converted {done} pages ({failed} failed) in {time.time() - t0:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
