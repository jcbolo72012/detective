"""Retrieval over the variable catalog (the "R" in RAG).

  python retrieve.py --build            # one-time: download NASA POWER's full monthly parameter catalog
  python retrieve.py "is it getting muggier"   # try a search

Index = curated entries in catalog.py (with plain-language keywords) + every POWER parameter in
power_params.json (if built). Search is TF-IDF (scikit-learn): no model download, runs instantly."""
import contextvars
import json
import os
import re
import sys

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import giovanni_catalog
from catalog import VARIABLES
from explorer import HOURLY

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
    allv.update(giovanni_catalog.load())  # keys start with "GIO:", so they never clash
    return allv


def is_giovanni(code):
    return str(code).startswith(giovanni_catalog.PREFIX)


# Per browser session (Streamlit runs each session in its own context): are Giovanni variables usable?
# The app sets this from the user's token on every run; it is never shared between users.
GIOVANNI_ON = contextvars.ContextVar("giovanni_on", default=False)


STOP = set("a an the is are was were be been it its this that these those of in on at to for from by with and or "
           "as than then there here how what when where why which who do does did has have had will would can "
           "could should getting get got become becoming more less much very over under above per during any "
           "average value values period surface earth meters meter summer winter spring fall autumn season "
           "year years month months since place changing change trend".split())


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
# hour-by-hour variants (about 1,100 near-duplicates) stay in ALL but are not searched
_ids = [k for k, v in ALL.items()
        if k in VARIABLES or is_giovanni(k) or not (HOURLY.search(k) or HOURLY.search(v['name']))]
# keywords repeated so plain-language terms outweigh boilerplate in long definitions
_docs = [stem(f"{v['name']} {v['name']} {v['definition']} {(' '.join(v['keywords']) + ' ') * 3}")
         for k, v in ((k, ALL[k]) for k in _ids)]
_vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"\S+")
_mat = _vec.fit_transform(_docs)


STEP_WEIGHT = {"monthly": 1.0, "weekly": 0.98, "daily": 0.97, "3-hourly": 0.92, "hourly": 0.88, "half-hourly": 0.82}


def _weight(code, gio_on):
    if code in VARIABLES:
        return 1.15  # curated: vetted definitions, known to work
    if is_giovanni(code):
        # usable only with a token; slightly below POWER so POWER (longer record, no login) wins ties.
        # Monthly products win over the same variable at hourly resolution (smaller, faster requests).
        if not gio_on:
            return 0.0
        v = ALL[code]
        w = 0.7 * STEP_WEIGHT.get(v.get("step") or "", 0.9)
        if v.get("region"):  # regional datasets (North America only, ...) lose to global ones
            w *= 0.9
        return w
    return 1.0


def search(question, k=5, min_score=0.03, giovanni=None):
    """Top-k catalog entries for a question: [(id, score), ...], best first. Empty if nothing matches.
    Giovanni variables are included only when `giovanni` (default: this session's GIOVANNI_ON) is true."""
    gio_on = GIOVANNI_ON.get() if giovanni is None else giovanni
    sims = linear_kernel(_vec.transform([stem(question)]), _mat).ravel()
    scored = sorted(((sims[i] * _weight(_ids[i], gio_on), _ids[i]) for i in range(len(_ids))), reverse=True)
    out, seen = [], set()
    for s, i in scored:
        if s < min_score or len(out) >= k:
            break
        name = ALL[i].get("short_name") if is_giovanni(i) else None
        if name and name in seen:  # same variable from another product/version: keep only the best one
            continue
        seen.add(name)
        out.append((i, round(float(s), 3)))
    return out


if __name__ == "__main__":
    if "--build" in sys.argv:
        build()
    else:
        q = " ".join(sys.argv[1:]) or "is it getting hotter"
        print(f"{len(ALL)} variables indexed")
        for i, s in search(q):
            print(f"{s:6.3f}  {i:20} {ALL[i]['name']}")
