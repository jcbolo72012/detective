"""Retrieval over the variable catalog (the "R" in RAG).

  python retrieve.py --build            # one-time: download NASA POWER's full monthly parameter catalog
  python retrieve.py "is it getting muggier"   # try a search

Index = curated entries in catalog.py (with plain-language keywords) + every POWER parameter in
power_params.json (if built) + Giovanni variables. Search is hybrid: TF-IDF (scikit-learn) and sentence
embeddings (bge-small via fastembed, vectors precomputed in catalog_embeddings.npz) merged by reciprocal
rank fusion. A hand-written lexicon (lexicon.py) adds everyday phrases per variable, misspelled words are
corrected against catalog words, and off-topic questions return nothing. Without the embedding file or model
it runs on TF-IDF alone.

  python retrieve.py --build-embeddings   # rerun after the catalog or lexicon.py changes
  python eval_retrieval.py                # score retrieval (add --heldout for the set not tuned against)"""
import contextvars
import difflib
import fnmatch
import functools
import hashlib
import json
import os
import re
import sys
import threading

import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import giovanni_catalog
from catalog import VARIABLES
from explorer import HOURLY
from lexicon import LEXICON

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


def _keywords(k):
    """(the entry's own keywords, every lexicon phrase whose key (id or fnmatch pattern) matches it)."""
    own = list(ALL[k]["keywords"])
    extra = [p for pat, phrases in LEXICON.items() if fnmatch.fnmatchcase(k, pat) for p in phrases if p not in own]
    return own, list(dict.fromkeys(extra))


_own, _extra = zip(*(_keywords(k) for k in _ids))
_kw = {k: [*o, *e] for k, o, e in zip(_ids, _own, _extra)}
LEX_REPEAT = 1  # lexicon phrases are repeated this many times in the TF-IDF text (own keywords: 3). More than 1 let long phrase lists drown out short ones, e.g. "temperature" for T2M


def _build_tfidf():
    global _vec, _mat
    docs = [stem(f"{ALL[k]['name']} {ALL[k]['name']} {ALL[k]['definition']} "
                 f"{(' '.join(o) + ' ') * 3}{(' '.join(e) + ' ') * LEX_REPEAT}") for k, o, e in zip(_ids, _own, _extra)]
    _vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"\S+")
    _mat = _vec.fit_transform(docs)


_build_tfidf()

ALIAS_BOOST = 0.15  # added to the keyword score when a whole keyword/lexicon phrase appears in the question
_aliases = {}
for _i, (_o, _e) in enumerate(zip(_own, _extra)):
    for _p in (*_o, *_e):
        _s = stem(_p)
        if _s:
            _aliases.setdefault(_s, set()).add(_i)

_WORDS = sorted({w for k in _ids for w in re.findall(r"[a-z]{4,}", f"{ALL[k]['name']} {' '.join(_kw[k])} "
                 f"{'' if is_giovanni(k) else ALL[k]['definition']}".lower()) if w not in STOP})
_WORD_SET = set(_WORDS)


def fix_typos(text):
    """Replace words the catalog has never seen with a close catalog word (warmr -> warmer, humdity -> humidity)."""
    out = []
    for w in text.split():
        if len(w) >= 5 and w.isalpha() and w not in STOP and w not in _WORD_SET and stem(w) not in _vec.vocabulary_:
            w = next(iter(difflib.get_close_matches(w, _WORDS, n=1, cutoff=0.84)), w)
        out.append(w)
    return " ".join(out)


# ---- embeddings: vectors are computed ahead of time (--build-embeddings) and committed; only the question is
# embedded at run time, with a small ONNX model (fastembed, no torch). Any failure falls back to TF-IDF alone.
EMB_FILE = os.path.join(HERE, "catalog_embeddings.npz")
EMB_MODEL = "BAAI/bge-small-en-v1.5"
USE_HYBRID = True
RRF_K, RRF_DEPTH = 60, 50
EMB_MIN = 0.58              # cosine below this is not a match, so off-topic questions can come back empty
EMB_WEIGHT_POWER = 0.75     # curated/Giovanni weights are softened for cosines, which are bunched together
ABSTAIN_COS, ABSTAIN_LEX, ABSTAIN_COS_WITH_LEX = 0.6, 0.15, 0.5


def _emb_texts():
    return [f"{ALL[k]['name']}. {ALL[k]['definition']}" + (f" Also: {', '.join(_kw[k])}." if _kw[k] else "") for k in _ids]


def _digest(texts):
    return hashlib.sha1("\x1f".join(_ids + texts).encode("utf-8")).hexdigest()


def build_embeddings():
    from fastembed import TextEmbedding
    texts = _emb_texts()
    vecs = np.array(list(TextEmbedding(EMB_MODEL).embed(texts)), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    np.savez_compressed(EMB_FILE, vecs=vecs.astype(np.float16), digest=np.array(_digest(texts)))
    print(f"Saved {len(texts)} vectors to {EMB_FILE}")


def _load_embeddings():
    """None if the file is missing or was built from a different catalog (then rebuild it)."""
    try:
        z = np.load(EMB_FILE)
        if str(z["digest"]) == _digest(_emb_texts()):
            return z["vecs"].astype(np.float32)
    except (OSError, KeyError, ValueError):
        pass
    return None


_EMB = _load_embeddings()
_model, _model_failed, _model_lock = None, False, threading.Lock()


@functools.lru_cache(maxsize=512)
def _embed(text):
    global _model, _model_failed
    if _model_failed:
        return None
    try:
        with _model_lock:
            if _model is None:
                from fastembed import TextEmbedding
                _model = TextEmbedding(EMB_MODEL)
            v = next(iter(_model.embed([text])))
        return v / np.linalg.norm(v)
    except Exception:  # offline, no disk, model missing: keep working on TF-IDF
        _model_failed = True
        return None


if _EMB is not None:  # load the model in the background so the first question doesn't wait for it
    threading.Thread(target=_embed, args=("warm up",), daemon=True).start()


STEP_WEIGHT = {"monthly": 1.0, "weekly": 0.98, "daily": 0.97, "3-hourly": 0.92, "hourly": 0.88, "half-hourly": 0.82}


GIO_BASE = 0.7
VARIANT_PENALTY = 0.92   # _SD/_MAX/_MIN/_AVG versions and wind components: prefer the base variable
VARIANT = re.compile(r"(_SD|_MAX|_MIN|_AVG|_SUM|_PRCNTL)$|^[UV]\d+M$")


def _weight(code, gio_on):
    if code in VARIABLES:
        return 1.15  # curated: vetted definitions, known to work
    if is_giovanni(code):
        # usable only with a token; slightly below POWER so POWER (longer record, no login) wins ties.
        # Monthly products win over the same variable at hourly resolution (smaller, faster requests).
        if not gio_on:
            return 0.0
        v = ALL[code]
        w = GIO_BASE * STEP_WEIGHT.get(v.get("step") or "", 0.9)
        if v.get("region"):  # regional datasets (North America only, ...) lose to global ones
            w *= 0.9
        return w
    return VARIANT_PENALTY if VARIANT.search(code) else 1.0


def search(question, k=5, min_score=0.03, giovanni=None, hybrid=None):
    """Top-k catalog entries for a question: [(id, score), ...], best first. Empty if nothing matches.
    Giovanni variables are included only when `giovanni` (default: this session's GIOVANNI_ON) is true.
    Hybrid (default when the embedding file and model are available): TF-IDF and embedding rankings are merged
    by reciprocal rank fusion. Score is then 0-1, where 1 means first in both rankings."""
    gio_on = GIOVANNI_ON.get() if giovanni is None else giovanni
    question = fix_typos(question)
    w = np.array([_weight(i, gio_on) for i in _ids])
    lex = linear_kernel(_vec.transform([stem(question)]), _mat).ravel()
    padded = f" {stem(question)} "
    for phrase, idx in _aliases.items():
        if f" {phrase} " in padded:
            lex[list(idx)] += ALIAS_BOOST
    lex *= w
    q_emb = _embed(question) if _EMB is not None and (USE_HYBRID if hybrid is None else hybrid) else None
    if q_emb is None:
        ranked = [(_ids[i], lex[i]) for i in np.argsort(-lex) if lex[i] >= min_score]
    else:
        cos = _EMB @ q_emb
        # off-topic questions: strong meaning match, or a keyword hit plus a fair meaning match
        if not (cos[w > 0].max() >= ABSTAIN_COS or (lex.max() >= ABSTAIN_LEX and cos[w > 0].max() >= ABSTAIN_COS_WITH_LEX)):
            return []
        emb = cos * w ** EMB_WEIGHT_POWER
        fused = np.zeros(len(_ids))
        for sims, keep in ((lex, lex >= min_score), (emb, (cos >= EMB_MIN) & (w > 0))):
            for rank, i in enumerate(i for i in np.argsort(-sims) if keep[i]):
                if rank >= RRF_DEPTH:
                    break
                fused[i] += 1 / (RRF_K + rank + 1)
        fused *= (RRF_K + 1) / 2
        ranked = [(_ids[i], fused[i]) for i in np.argsort(-fused) if fused[i] > 0]
    out, seen = [], set()
    for i, s in ranked:
        if len(out) >= k:
            break
        name = ALL[i].get("short_name") if is_giovanni(i) else None
        if name and name in seen:  # same variable from another product/version: keep only the best one
            continue
        seen.add(name)
        out.append((i, round(float(s), 3)))
    return out


def variable_text(question):
    """The question with place names and years removed, so the variable search sees only what is being measured
    ("Is Sahara dust rising near Cairo since 1990" must not match catalog text that mentions the Sahara or 1990)."""
    import geo
    words = geo.fold(question).split()
    drop = [False] * len(words)
    for pl in geo.find_places(question, max_alts=0):
        key = pl["text"].split()
        for i in range(len(words) - len(key) + 1):
            if words[i:i + len(key)] == key:
                drop[i:i + len(key)] = [True] * len(key)
    return " ".join(w for w, d in zip(words, drop) if not d and not re.fullmatch(r"(19|20)\d\d", w))


def search_question(question, k=5, min_score=0.03, giovanni=None, hybrid=None):
    """search() for a natural-language question: places are stripped first. Falls back to the whole
    question if stripping leaves nothing to search on."""
    text = variable_text(question)
    return (text and search(text, k, min_score, giovanni, hybrid)) or search(question, k, min_score, giovanni, hybrid)


if __name__ == "__main__":
    if "--build" in sys.argv:
        build()
    elif "--build-embeddings" in sys.argv:
        build_embeddings()
    else:
        q = " ".join(sys.argv[1:]) or "is it getting hotter"
        print(f"{len(ALL)} variables indexed; {'hybrid' if _EMB is not None and USE_HYBRID else 'TF-IDF only'}")
        for i, s in search_question(q):
            print(f"{s:6.3f}  {i:20} {ALL[i]['name']}")
