"""Score variable retrieval against eval/retrieval_cases.json.

  python eval/eval_retrieval.py              # compare TF-IDF, TF-IDF without places, and hybrid (needs catalog_embeddings.npz)
  python eval/eval_retrieval.py -v           # also list every miss
  python eval/eval_retrieval.py --only multi # one tag

Each case lists groups of acceptable catalog ids (fnmatch patterns allowed). One group = one variable the
question asks about, so a two-variable question has two groups.
  top1   the best hit is acceptable for some group
  cov@3  every group has an acceptable id in the top 3     (cov@5: top 5)
Cases tagged "none" have no right answer: they pass when nothing is returned."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse
import collections
import fnmatch
import json
import os
import sys

from planets.query import retrieve

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = os.path.join(HERE, "retrieval_cases.json")
HELDOUT = os.path.join(HERE, "retrieval_cases_heldout.json")


def _ok(hit, group):
    return any(fnmatch.fnmatchcase(hit, pat) for pat in group)


def score(case, search):
    hits = [h for h, _ in search(case["q"], k=5, giovanni=bool(case.get("gio")))]
    groups = case["g"]
    if not groups:
        return dict(top1=not hits, cov3=not hits, cov5=not hits, hits=hits)
    cov = lambda n: all(any(_ok(h, g) for h in hits[:n]) for g in groups)
    return dict(top1=bool(hits) and any(_ok(hits[0], g) for g in groups), cov3=cov(3), cov5=cov(5), hits=hits)


def split_check(cases):
    """Does nl.question_variables find one variable per group? Single-variable questions must not be split."""
    from planets.query import nl
    multi, single, rows = [], [], []
    for c in cases:
        retrieve.GIOVANNI_ON.set(bool(c.get("gio")))
        got = [v for v, _ in nl.question_variables(c["q"])]
        groups = c["g"]
        if len(groups) >= 2:
            free, ok = list(got), True
            for g in groups:  # each group needs its own detected variable
                m = next((v for v in free if _ok(v, g)), None)
                ok = ok and m is not None
                if m is not None:
                    free.remove(m)
            multi.append(ok)
            if not ok:
                rows.append(f"  [missed split] {c['q']}\n      got: {', '.join(v.replace('GIO:', '') for v in got) or '(nothing)'}")
        elif groups:
            single.append(len(got) >= 2)
            if len(got) >= 2:
                rows.append(f"  [wrongly split] {c['q']}\n      got: {', '.join(v.replace('GIO:', '') for v in got)}")
    retrieve.GIOVANNI_ON.set(False)
    return multi, single, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true", help="list misses")
    ap.add_argument("--only", help="run only cases with this tag")
    ap.add_argument("--heldout", action="store_true",
                    help="use eval/retrieval_cases_heldout.json: written before the lexicon, never tuned against")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    with open(HELDOUT if a.heldout else CASES, encoding="utf-8") as f:
        cases = [c for c in json.load(f) if not a.only or a.only in c["t"]]
    engines = [("tfidf", lambda q, **kw: retrieve.search(q, hybrid=False, **kw)),
               ("tfidf-np", lambda q, **kw: retrieve.search_question(q, hybrid=False, **kw)),
               ("hybrid", lambda q, **kw: retrieve.search_question(q, hybrid=True, **kw))]
    if retrieve._EMB is None:
        sys.exit("No catalog_embeddings.npz for this catalog: run  python -m planets.query.retrieve --build-embeddings")
    res = {name: [score(c, fn) for c in cases] for name, fn in engines}

    def table(title, pick):
        print(f"\n{title}")
        print(f"{'':12}{'n':>4}" + "".join(f"  {name + ' ' + m:>15}" for name, _ in engines for m in ("top1", "cov@3", "cov@5")))
        for label, idx in pick:
            if not idx:
                continue
            row = f"{label:12}{len(idx):>4}"
            for name, _ in engines:
                for m in ("top1", "cov3", "cov5"):
                    row += f"  {100 * sum(res[name][i][m] for i in idx) / len(idx):>14.0f}%"
            print(row)

    tags = collections.OrderedDict()
    for i, c in enumerate(cases):
        for t in c["t"]:
            tags.setdefault(t, []).append(i)
    table("By tag (percent of cases)", [("ALL", list(range(len(cases))))] + sorted(tags.items()))
    multi, single, split_rows = split_check(cases)
    if multi:
        print(f"\nSplitting a question into its variables: found every variable in {sum(multi)} of {len(multi)} "
              f"multi-variable questions; wrongly split {sum(single)} of {len(single)} single-variable ones")
    if a.verbose:
        print("\n".join(split_rows))

    if a.verbose:
        for name, _ in engines:
            print(f"\nMisses at cov@5, engine={name}")
            for c, r in zip(cases, res[name]):
                if not r["cov5"]:
                    print(f"  [{','.join(c['t'])}] {c['q']}\n      got: {', '.join(h.replace('GIO:', '') for h in r['hits']) or '(nothing)'}")
        print("\nTop-1 misses (engine=hybrid)")
        for c, r in zip(cases, res["hybrid"]):
            if not r["top1"] and r["cov5"]:
                print(f"  [{','.join(c['t'])}] {c['q']}\n      got: {', '.join(h.replace('GIO:', '') for h in r['hits'][:3])}")


if __name__ == "__main__":
    main()
