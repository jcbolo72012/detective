"""Language layer. The AI only (1) turns a question into parameters and (2) words the computed numbers.
It never calculates. Works with no API key too (keyword parser + template text).

Set ONE of:  ANTHROPIC_API_KEY  or  OPENAI_API_KEY   (optional: LLM_MODEL to override the model name)"""
import json
import os
import re

import geo
from catalog import PLACES, RELATED
from catalog import VARIABLES as CURATED
from power import LAST_FULL_YEAR
from retrieve import ALL as VARIABLES, search



def _load_env(path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")):
    """Read KEY=value lines from .env next to this file. Real environment variables win; blanks are ignored."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and v and not os.environ.get(k):
                os.environ[k] = v


_load_env()

SEASON_WORDS = {"summer": "JJA", "winter": "DJF", "spring": "MAM", "fall": "SON", "autumn": "SON"}
FLIP = {"JJA": "DJF", "DJF": "JJA", "MAM": "SON", "SON": "MAM"}


_AI_CALLS = {"day": None, "n": 0}


def _cap_reached():
    """Process-wide daily cap on AI calls, so a public link can't run up the bill. Resets at UTC midnight."""
    import datetime
    today = datetime.date.today().isoformat()
    if _AI_CALLS["day"] != today:
        _AI_CALLS.update(day=today, n=0)
    try:
        cap = int(os.getenv("AI_DAILY_CAP", "300"))
    except ValueError:
        cap = 300
    return _AI_CALLS["n"] >= cap


def llm_available():
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")) and not _cap_reached()


def _llm(system, user, max_tokens=500):
    if _cap_reached():
        raise ValueError("The AI daily limit was reached, so the app is using its no-AI mode for now.")
    _AI_CALLS["n"] += 1
    if os.getenv("ANTHROPIC_API_KEY"):
        import anthropic
        msg = anthropic.Anthropic().messages.create(
            model=os.getenv("LLM_MODEL", "claude-haiku-5-5"), max_tokens=max_tokens + 2000,  # room for thinking
            system=system, messages=[{"role": "user", "content": user}])
        # the reply can start with a thinking block; keep only the text blocks
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        if not text.strip():
            raise ValueError(f"The AI returned no text (stop reason: {msg.stop_reason}). Try again.")
        return text
    from openai import OpenAI
    out = OpenAI().chat.completions.create(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return out.choices[0].message.content


# ------------------------------------------------------------ question -> parameters
def _giovanni_hint(q):
    """If only a (locked) Giovanni variable matches, say so instead of a generic 'no match'."""
    if not q:
        return None
    try:
        hits = search(q, k=1, giovanni=True)
    except TypeError:
        return None
    if hits and str(hits[0][0]).startswith("GIO:"):
        return (f"That sounds like **{VARIABLES[hits[0][0]]['name']}**, which comes from NASA Giovanni. Add your "
                "free Earthdata token under 'NASA Giovanni' in the sidebar to investigate it.")
    return None


def validate(p, question=None):
    """Never trust the model: clamp everything to what the catalog and API support."""
    if p.get("variable") not in VARIABLES:
        hint = _giovanni_hint(question)
        if hint:
            raise ValueError(hint)
        raise ValueError("I couldn't match that to a variable in the catalog. Try words like temperature, rain, "
                         "soil moisture, humidity, wind or sunlight.")
    lat, lon = float(p["lat"]), float(p["lon"])
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("Couldn't work out where that place is.")
    season = p.get("season") or None
    if season not in (None, "DJF", "MAM", "JJA", "SON"):
        season = None
    first, last = 1981, LAST_FULL_YEAR
    v = VARIABLES[p["variable"]]
    if v.get("provider") == "giovanni":  # each Giovanni dataset has its own record (GLDAS 2.0: 1948-2014, OMI: 2005-)
        first = max(1948, int(v.get("first_year") or 1980))
        last = min(LAST_FULL_YEAR, int(v.get("last_year") or LAST_FULL_YEAR))
        if not p.get("start_year"):
            default = first
            if v.get("step") in ("hourly", "half-hourly"):  # big downloads: default to the last 20 years
                default = max(first, last - 19)
            p = {**p, "start_year": default}
    span = min(10, max(1, last - first))
    start = min(max(int(p.get("start_year") or first), first), last - span)
    end = min(max(int(p.get("end_year") or last), start + span), last)
    return dict(variable=p["variable"], place=str(p.get("place") or f"{lat:.2f}, {lon:.2f}"),
                lat=lat, lon=lon, season=season, start_year=start, end_year=end)


def parse_keywords(q):
    """No-AI fallback: keyword scoring against the catalog."""
    ql = q.lower()
    hits = search(q, k=5)
    var = hits[0][0] if hits else None
    found = geo.find_places(q)
    if not found:
        raise ValueError(no_place_message())
    pl = found[0]
    place, lat, lon = pl["label"], pl["lat"], pl["lon"]
    season = next((c for w, c in SEASON_WORDS.items() if w in ql), None)
    if season and lat < 0:
        season = FLIP[season]
    yrs = [int(y) for y in re.findall(r"\b(19[89]\d|20[0-2]\d)\b", ql)]
    p = validate(dict(variable=var, place=place, lat=lat, lon=lon, season=season,
                      start_year=min(yrs) if yrs else None, end_year=max(yrs) if len(yrs) > 1 else None), q)
    return {**p, "candidates": hits, "place_alts": pl["alts"]}


def no_place_message(n=1):
    need = "a place" if n == 1 else "two places"
    if geo.AVAILABLE:
        return (f"I couldn't find {need} in that question. Name a city, country or US state "
                "(for example Chicago, Kenya or Texas), or use the menus.")
    return (f"I couldn't find {need} in that question. Without the city list installed (run `uv sync`) "
            "I only know: " + ", ".join(PLACES))


def parse_question(q):
    """Returns (params, used_ai)."""
    if not llm_available():
        return parse_keywords(q), False
    hits = search(q, k=6)
    if not hits:
        raise ValueError("Nothing in the data catalog matches that question. Try different words.")
    cat = "\n".join(f'- {k}: {VARIABLES[k]["name"]} [{VARIABLES[k]["units"]}] ({VARIABLES[k]["definition"][:200]})'
                    for k, _ in hits)
    system = (
        "You translate a question about environmental change into query parameters. Reply with ONLY a JSON "
        'object: {"variable": <id from the list>, "place": <short place name>, "lat": <number>, "lon": <number>, '
        '"season": null|"DJF"|"MAM"|"JJA"|"SON", "start_year": <int or null>, "end_year": <int or null>}.\n'
        "These candidates were retrieved from the NASA data catalog. Pick the single best variable id "
        "from this list ONLY, or null if none fits:\n" + cat + "\n"
        "lat/lon: a representative point for the place (decimal degrees, west/south negative). "
        "season: only if the question names one; convert to the months that season has AT THAT PLACE "
        "(southern-hemisphere summer is DJF). Years: only if stated. Do not answer the question itself.")
    txt = _llm(system, q, 300)
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        raise ValueError("The AI didn't return parameters. Try rephrasing, or use the menus.")
    p = validate(json.loads(m.group(0)))
    if p["variable"] not in dict(hits):
        raise ValueError("The AI picked a variable outside the retrieved candidates. Try rephrasing.")
    return {**p, "candidates": hits, "place_alts": []}, True


# ------------------------------------------------------------ numbers -> words
def explain_template(ctx):
    m, v = ctx["metrics"], ctx["variable"]
    when = f"{ctx['season']} " if ctx.get("season") else ""
    t = (f"**{v['name']}** at **{ctx['place']}** has been **{m['direction']}** by about "
         f"**{abs(m['sen_slope_per_decade']):.3g} {v['units']} per decade** "
         f"({when}{m['start'][:4]}–{m['end'][:4]}; straight-line fit: {m['ols_slope_per_decade']:+.3g}, "
         f"95% range {m['ols_ci95_low']:+.3g} to {m['ols_ci95_high']:+.3g}). ")
    if m["significant"]:
        t += ("Both tests (t-test and Mann-Kendall) say this is **statistically significant**: "
              "a trend this strong would be unlikely to appear by chance in random year-to-year wiggles. ")
    elif m["tests_agree"]:
        t += ("This is **not statistically significant**: the ups and downs from year to year are big enough "
              "that a trend this size could be chance. A longer record might settle it. ")
    else:
        t += "The two tests **disagree**, so treat this as a borderline, 'maybe' trend. "
    if m["lag1_autocorr_resid"] > 0.3:
        t += (f"Caution: neighbouring months are strongly related (autocorrelation {m['lag1_autocorr_resid']:.2f}), "
              "which makes the p-values look better than they are. ")
    if ctx.get("neighbors"):
        n = ctx["neighbors"]
        t += (f"Nearby check: {n['same_direction']} of {n['n']} surrounding points trend the same way "
              f"({n['significant']} significantly). Neighbours share weather, so this is a consistency check, "
              "not extra independent proof. ")
    return t


# ------------------------------------------------------------ next clues
GENERIC_CLUES = [("T2M", "to see whether air temperature is changing at the same place over the same period."),
                 ("PRECTOTCORR", "to see whether rainfall is changing alongside it.")]


def clues_fallback(var_id):
    pairs = RELATED.get(var_id) or GENERIC_CLUES
    return [{"variable": v, "why": w} for v, w in pairs if v in CURATED and v != var_id][:2]


def suggest_clues(ctx):
    """ctx as for explain() plus variable_id. Returns (up to 2 x {"variable": id, "why": str}, used_ai)."""
    var_id = ctx.get("variable_id")
    if not llm_available():
        return clues_fallback(var_id), False
    options = "\n".join(f'- {k}: {v["name"]} ({v["definition"]})' for k, v in CURATED.items() if k != var_id)
    system = (
        "You help a student investigate environmental change like a detective. You are given a question, a place, "
        "the variable just analysed and its computed trend statistics. Suggest the 2 most useful variables to check "
        "NEXT at the same place, chosen ONLY from this list:\n" + options + "\n"
        'Reply with ONLY a JSON list: [{"variable": <id from the list>, "why": <one sentence>}, ...]. '
        "Each 'why' must be a hypothesis to check, starting with 'to see whether', in plain language. "
        "Never state a cause as fact, never predict what the check will show, and do not include any numbers.")
    payload = {k: ctx[k] for k in ("question", "place", "season", "variable", "metrics") if ctx.get(k) is not None}
    try:
        txt = _llm(system, json.dumps(payload, default=str), 400)
        items = json.loads(re.search(r"\[.*\]", txt, re.S).group(0))
        out = []
        for it in items:
            vid, why = it.get("variable"), str(it.get("why") or "").strip()
            if vid in CURATED and vid != var_id and why and vid not in [o["variable"] for o in out]:
                out.append({"variable": vid, "why": why[:300]})
        if out:
            return out[:2], True
    except Exception:
        pass  # any failure: fall back to the hand-written suggestions
    return clues_fallback(var_id), False


def explain(ctx):
    """ctx = {question, place, season, variable{name,units,definition}, metrics, neighbors?}. Returns (text, used_ai)."""
    if not llm_available():
        return explain_template(ctx), False
    system = (
        "You are a science communicator for high-school and college students. You are given the user's question "
        "and a JSON of statistics that were ALREADY computed by code. Write 4-6 sentences: (1) direct answer to "
        "the question, (2) how much it is changing per decade with units and the 95% range, (3) whether it is "
        "statistically significant and what that means in plain words, (4) any caveats present in the JSON "
        "(autocorrelation above 0.3, tests disagreeing, short record, neighbour disagreement; neighbours are not "
        "independent evidence), (5) one 'next clue': another variable worth checking and why. "
        "RULES: use ONLY numbers from the JSON, never compute or invent numbers, do not claim a cause as fact "
        "(say 'one possible driver'), do not extrapolate into the future.")
    payload = {k: ctx[k] for k in ("question", "place", "season", "variable", "metrics", "neighbors") if ctx.get(k) is not None}
    try:
        return _llm(system, json.dumps(payload, default=str), 500), True
    except Exception as e:  # never let the explanation break the demo
        return explain_template(ctx) + f"\n\n_(AI explanation unavailable: {e})_", False


# ------------------------------------------------------------ comparing two trends
COMPARE_RE = re.compile(r"\b(vs\.?|versus|compared? (to|with)|\w+er than|(more|less) \w+ than|while|whereas)\b", re.I)


def looks_like_comparison(q):
    return bool(COMPARE_RE.search(q or ""))


def _pair(a, b, season, start, end, hits):
    pa = validate(dict(**a, season=season, start_year=start, end_year=end))
    pb = validate(dict(**b, season=season, start_year=pa["start_year"], end_year=pa["end_year"]))
    return dict(a=pa, b=pb, season=pa["season"], start_year=pa["start_year"], end_year=pa["end_year"], candidates=hits)


def parse_compare_keywords(q):
    ql = q.lower()
    hits = search(q, k=6)
    places = geo.find_places(q)
    if not hits:
        raise ValueError("I couldn't match that to a variable in the catalog. Try the menus below.")
    if len(places) < 2:
        raise ValueError(no_place_message(2))
    season = next((c for w, c in SEASON_WORDS.items() if w in ql), None)
    if season and places[0]["lat"] < 0:
        season = FLIP[season]
    yrs = [int(y) for y in re.findall(r"\b(19[89]\d|20[0-2]\d)\b", ql)]
    mk = lambda m: dict(variable=hits[0][0], place=m["label"], lat=m["lat"], lon=m["lon"])
    return _pair(mk(places[0]), mk(places[1]), season, min(yrs) if yrs else None,
                 max(yrs) if len(yrs) > 1 else None, hits)


def parse_compare(q):
    """Comparison question -> ({a, b, season, start_year, end_year, candidates}, used_ai)."""
    if not llm_available():
        return parse_compare_keywords(q), False
    hits = search(q, k=8)
    if not hits:
        raise ValueError("Nothing in the data catalog matches that question. Try different words.")
    cat = "\n".join(f'- {k}: {VARIABLES[k]["name"]} [{VARIABLES[k]["units"]}] ({VARIABLES[k]["definition"][:160]})'
                    for k, _ in hits)
    system = (
        "You translate a question that COMPARES two environmental trends into query parameters. Reply with ONLY "
        'a JSON object: {"a": {"variable": <id>, "place": <short name>, "lat": <number>, "lon": <number>}, '
        '"b": {"variable": <id>, "place": <short name>, "lat": <number>, "lon": <number>}, '
        '"season": null|"DJF"|"MAM"|"JJA"|"SON", "start_year": <int or null>, "end_year": <int or null>}.\n'
        "a and b are the two things compared, in the order the question names them: usually the same variable at "
        "two places, sometimes two variables at one place. Variable ids must come from this retrieved list ONLY:\n"
        + cat + "\nlat/lon: a representative point (decimal degrees, west/south negative). season: only if the "
        "question names one, as the months at place a. Years: only if stated. Do not answer the question itself.")
    txt = _llm(system, q, 400)
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        raise ValueError("The AI didn't return parameters. Try rephrasing, or use the menus below.")
    j = json.loads(m.group(0))
    out = _pair({k: j["a"].get(k) for k in ("variable", "place", "lat", "lon")},
                {k: j["b"].get(k) for k in ("variable", "place", "lat", "lon")},
                j.get("season"), j.get("start_year"), j.get("end_year"), hits)
    for side in ("a", "b"):
        if out[side]["variable"] not in dict(hits):
            raise ValueError("The AI picked a variable outside the retrieved candidates. Try rephrasing.")
    return out, True


def explain_compare(ctx):
    """ctx = {question, season, headline, a{label,units,metrics}, b{...}, comparison{...}}. Returns (text, used_ai)."""
    import compare as _cmp
    if not llm_available():
        return _cmp.explain_template(ctx), False
    system = (
        "You are a science communicator for high-school and college students. You are given a JSON comparing two "
        "trends, with statistics ALREADY computed by code. Write 4-6 sentences: (1) the direct answer, consistent "
        "with the given headline; (2) how fast each is changing per decade, with units; (3) what the test on the "
        "difference (A minus B) says, if present: it is the direct test of whether one is changing faster; "
        "(4) caveats present in the JSON (a trend that is not significant, tests disagreeing, autocorrelation "
        "above 0.3, only one side significant while the difference is not); (5) one thing worth checking next. "
        "RULES: use ONLY numbers from the JSON, never compute or invent numbers, never claim a cause as fact, "
        "do not extrapolate into the future. The 'correlation' block (trends removed first, p adjusted for autocorrelation) describes shared short-term swings only; never say one causes the other, and treat a delay (best_lag) as exploratory.")
    try:
        return _llm(system, json.dumps(ctx, default=str), 500), True
    except Exception as e:
        return _cmp.explain_template(ctx) + f"\n\n_(AI explanation unavailable: {e})_", False
