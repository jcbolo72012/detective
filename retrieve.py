"""Retrieval over the variable catalog (the "R" in RAG).

  python retrieve.py --build            # one-time: download NASA POWER's full monthly parameter catalog
  python retrieve.py "is it getting muggier"   # try a search

Index = curated entries in catalog.py (with plain-language keywords) + every POWER parameter in
power_params.json (if built). Search is TF-IDF (scikit-learn): no model download, runs instantly."""
import json
import os
import re
import sys

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from catalog import VARIABLES

HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS_FILE = os.path.join(HERE, "power_params.json")
MANAGER_URL = "https://power.larc.nasa.gov/api/system/manager/parameters"


def build(community="AG"):
    r = requests.get(MANAGER_URL, params=dict(temporal="MONTHLY", community=community), timeout=90)
    r.raise_for_status()
    raw, out = r.json(), {}
    for code, d in raw.items():
        if not isinstance(d, dict):
            continue
        name = d.get("name") or d.get("longname") or code
        out[code] = dict(name=str(name), units=str(d.get("units") or ""),
                         definition=str(d.get("definition") or name),
                         source=f"NASA POWER ({d.get('source') or d.get('type') or 'MERRA-2 / CERES'})")
    if not out:
        raise RuntimeError("Unexpected response shape; first 300 chars: " + json.dumps(raw)[:300])
    with open(PARAMS_FILE, "w") as f:
        json.dump(out, f, indent=1)
    print(f"Saved {len(out)} parameters to {PARAMS_FILE}")


def load_all():
    """Merged catalog: {id: {name, units, definition, source, keywords}}. Curated entries win."""
    allv = {}
    if os.path.exists(PARAMS_FILE):
        with open(PARAMS_FILE) as f:
            allv.update({k: {**v, "keywords": []} for k, v in json.load(f).items()})
    allv.update(VARIABLES)
    return allv


STOP = set("a an the is are was were be been it its this that these those of in on at to for from by with and or "
           "as than then there here how what when where why which who do does did has have had will would can "
           "could should getting get got become becoming more less much very over under above per during any "
           "average value values period surface earth meters meter".split())


def stem(text):
    """Crude stemmer so hotter/hot, drier/dry/drying, windier/windy, cloudier/cloudy share a token."""
    out = []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w in STOP:
            continue
        for suf in ("iest", "ier", "est", "ing", "ed", "er", "es", "s"):
            if w.endswith(suf) and len(w) - len(suf) >= 2 and not w.endswith("ss"):
                w = w[:-len(suf)]
                break
        if len(w) > 3 and w[-1] in "yie":
            w = w[:-1]
        if len(w) > 2 and w[-1] == w[-2]:
            w = w[:-1]
        out.append(w)
    return " ".join(out)


ALL = load_all()
_ids = list(ALL)
# keywords repeated so plain-language terms outweigh boilerplate in long definitions
_docs = [stem(f"{v['name']} {v['name']} {v['definition']} {(' '.join(v['keywords']) + ' ') * 3}")
         for k, v in ALL.items()]
_vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"\S+")
_mat = _vec.fit_transform(_docs)


def search(question, k=5, min_score=0.03):
    """Top-k catalog entries for a question: [(id, score), ...], best first. Empty if nothing matches."""
    sims = linear_kernel(_vec.transform([stem(question)]), _mat).ravel()
    # tiny bonus for curated entries: they have vetted definitions and are known to work
    scored = sorted(((sims[i] * (1.15 if _ids[i] in VARIABLES else 1.0), _ids[i]) for i in range(len(_ids))),
                    reverse=True)
    return [(i, round(float(s), 3)) for s, i in scored[:k] if s >= min_score]


if __name__ == "__main__":
    if "--build" in sys.argv:
        build()
    else:
        q = " ".join(sys.argv[1:]) or "is it getting hotter"
        print(f"{len(ALL)} variables indexed")
        for i, s in search(q):
            print(f"{s:6.3f}  {i:20} {ALL[i]['name']}")
