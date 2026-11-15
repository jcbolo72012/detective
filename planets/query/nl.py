"""Language layer. The AI only (1) turns a question into parameters and (2) words the computed numbers.
It never calculates. Works with no API key too (keyword parser + template text).

Set ONE of:  ANTHROPIC_API_KEY  or  OPENAI_API_KEY   (optional: LLM_MODEL to override the model name)"""
import contextvars
import json
import os
import re

from planets import config
from planets.query import geo
from planets.sources import providers
from planets.query.catalog import PLACES, RELATED
from planets.query.catalog import VARIABLES as CURATED
from planets.sources.power import LAST_FULL_YEAR
from planets.query.retrieve import ALL as VARIABLES, search_question



config.load_env()

SEASON_WORDS = {"summer": "JJA", "winter": "DJF", "spring": "MAM", "fall": "SON", "autumn": "SON"}
SEASON_RE = re.compile(r"\b(" + "|".join(SEASON_WORDS) + r")s?\b")  # whole words only: "rainfall" is not "fall"


def season_in(ql):
    m = SEASON_RE.search(ql)
    return SEASON_WORDS[m.group(1)] if m else None
FLIP = {"JJA": "DJF", "DJF": "JJA", "MAM": "SON", "SON": "MAM"}


_AI_CALLS = {"day": None, "n": 0}

# A visitor's own Claude key. Per script run (each Streamlit session runs in its own context), never in os.environ,
# a cache key, a log line or an error message, so it cannot reach another visitor. The app sets it on every run.
USER_API_KEY = contextvars.ContextVar("user_api_key", default="")


def set_user_key(raw):
    """Use `raw` (a pasted Anthropic key) for this session's AI calls. Returns True if it looks like a key."""
    key = re.sub(r"\s+", "", raw or "")
    USER_API_KEY.set(key if key.startswith("sk-ant-") else "")
    return bool(USER_API_KEY.get())


def ai_mode():
    """'user' (the visitor's own key), 'shared' (the app's key, under the daily cap) or '' (no AI)."""
    return "user" if USER_API_KEY.get() else ("shared" if llm_available() else "")


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
    if USER_API_KEY.get():  # their own key: their own bill, so the shared daily cap does not apply
        return True
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")) and not _cap_reached()


def check_user_key():
    """(ok, message) for this session's own key, using one tiny request."""
    if not USER_API_KEY.get():
        return False, "That doesn't look like an Anthropic key (they start with sk-ant-)."
    try:
        _llm("Reply with the single word OK.", "ping", 5)
        return True, "Key works. AI wording is on for your session."
    except ValueError as e:
        return False, str(e)


def _claude(system, user, max_tokens, api_key=None):
    import anthropic
    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    msg = client.messages.create(
        model=os.getenv("LLM_MODEL", "claude-haiku-5-5"), max_tokens=max_tokens + 2000,  # room for thinking
        system=system, messages=[{"role": "user", "content": user}])
    # the reply can start with a thinking block; keep only the text blocks
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    if not text.strip():
        raise ValueError(f"The AI returned no text (stop reason: {msg.stop_reason}). Try again.")
    return text


def _llm(system, user, max_tokens=500):
    key = USER_API_KEY.get()
    if key:
        try:
            return _claude(system, user, max_tokens, api_key=key)
        except ValueError:
            raise
        except Exception as e:  # say what kind of failure, never echo the exception text (it could quote the key)
            kind = type(e).__name__
            hint = " Check the key in the sidebar." if "Authentication" in kind or "Permission" in kind else ""
            raise ValueError(f"Your Claude key could not be used ({kind}).{hint}") from None
    if _cap_reached():
        raise ValueError("The AI daily limit was reached, so the app is using its no-AI mode for now.")
    _AI_CALLS["n"] += 1
    if os.getenv("ANTHROPIC_API_KEY"):
        return _claude(system, user, max_tokens)
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
        hits = search_question(q, k=1, giovanni=True)
    except TypeError:
        return None
    if hits and providers.needs_token(hits[0][0]):
        return (f"That sounds like **{VARIABLES[hits[0][0]]['name']}**, which comes from {providers.label_of(hits[0][0])}. "
                "Add your free Earthdata token under 'NASA Giovanni' in the sidebar to investigate it.")
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
    if v.get("provider") in ("giovanni", "podaac"):  # each dataset has its own record (GLDAS 2.0: 1948-2014, OMI: 2005-)
        first = max(1948, int(v.get("first_year") or 1980))
        last = min(LAST_FULL_YEAR, int(v.get("last_year") or LAST_FULL_YEAR))
        if not p.get("start_year"):
            default = first
            if v.get("step") in ("hourly", "half-hourly"):  # big downloads: default to the last 20 years
                default = max(first, last - 19)
            if v.get("default_years"):  # many small requests per month of record: keep the default window short
                default = max(first, last - int(v["default_years"]) + 1)
            p = {**p, "start_year": default}
    span = min(10, max(1, last - first))
    start = min(max(int(p.get("start_year") or first), first), last - span)
    end = min(max(int(p.get("end_year") or last), start + span), last)
    return dict(variable=p["variable"], place=str(p.get("place") or f"{lat:.2f}, {lon:.2f}"),
                lat=lat, lon=lon, season=season, start_year=start, end_year=end)


def parse_keywords(q):
    """No-AI fallback: keyword scoring against the catalog."""
    ql = q.lower()
    hits = search_question(q, k=5)
    var = hits[0][0] if hits else None
    found = geo.find_places(q)
    if not found:
        raise ValueError(no_place_message())
    pl = found[0]
    place, lat, lon = pl["label"], pl["lat"], pl["lon"]
    season = season_in(ql)
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
    hits = search_question(q, k=6)
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
    try:
        txt = _llm(system, q, 300)
    except ValueError:
        if USER_API_KEY.get():  # their key failed: still answer, with the keyword parser
            return parse_keywords(q), False
        raise
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        raise ValueError("The AI didn't return parameters. Try rephrasing, or use the menus.")
    p = validate(json.loads(m.group(0)), q)
    if p["variable"] not in dict(hits):
        raise ValueError("The AI picked a variable outside the retrieved candidates. Try rephrasing.")
    return {**p, "candidates": hits, "place_alts": []}, True


# ------------------------------------------------------------ numbers -> words
def explain_template(ctx):
    m, v = ctx["metrics"], ctx["variable"]
    when = f"{ctx['season']} " if ctx.get("season") else ""
    opener = ("**Case cracked, Detective.** " if m["significant"] else
              "**The trail has gone cold, for now.** " if m["tests_agree"] else "**A hung jury.** ")
    t = opener + (f"**{v['name']}** at **{ctx['place']}** has been **{m['direction']}** by about "
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
    if m.get("seasonal_cycle_kept"):
        t += ("The seasonal cycle was **kept** in this test, so the swing between summer and winter counts as "
              "noise; that makes a real trend harder to detect than with the cycle removed. ")
    if m["lag1_autocorr_resid"] > 0.3:
        t += (f"Caution: neighbouring months are strongly related (autocorrelation {m['lag1_autocorr_resid']:.2f}), "
              "which makes the p-values look better than they are. ")
    dc = ctx.get("data_check")
    if dc and dc["level"] != "green":
        t += f"Data check: {'; '.join(dc['issues'])}. "
    if ctx.get("neighbors"):
        n = ctx["neighbors"]
        t += (f"Canvassing the neighbours: {n['same_direction']} of {n['n']} surrounding points trend the same way "
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
        "You are the Planeteer, sidekick to a student who is the detective investigating environmental change. "
        "You are given a question, a place, "
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
        "You are the Planeteer, a friendly science sidekick to a student detective (high-school or college). "
        "Address them as Detective once, with at most one light, playful line; clarity comes first. "
        "You are given the user's question "
        "and a JSON of statistics that were ALREADY computed by code. Write 4-6 sentences: (1) direct answer to "
        "the question, (2) how much it is changing per decade with units and the 95% range, (3) whether it is "
        "statistically significant and what that means in plain words, (4) any caveats present in the JSON "
        "(autocorrelation above 0.3, tests disagreeing, short record, neighbour disagreement; neighbours are not "
        "independent evidence; if metrics.seasonal_cycle_kept is true, say the summer-winter swing was left in and "
        "counts as noise, which makes the test weaker; if data_check.level is not 'green', mention its issues), (5) one 'next clue': another variable worth checking and why. "
        "RULES: use ONLY numbers from the JSON, never compute or invent numbers, do not claim a cause as fact "
        "(say 'one possible driver'), do not extrapolate into the future.")
    payload = {k: ctx[k] for k in ("question", "place", "season", "variable", "metrics", "neighbors", "data_check")
               if ctx.get(k) is not None}
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


# Words that join or relate two things being asked about: "hotter AND drier", "sunshine MEAN higher temperatures".
CLAUSE_SPLIT = re.compile(r"\s*(?:[,;&]|\b(?:and|plus|but|while|whereas|as|with|or|versus|vs\.?|means?|causes?|"
                          r"affects?|drives?|leads? to|linked to|related to|correlates? with)\b)\s*", re.I)


def question_variables(q, max_vars=3):
    """Variables a question asks about, in order: [(variable id, place dict or None), ...].
    The question is cut at joining words and each piece is searched on its own, so "hotter and drier" gives
    air temperature and precipitation. Pieces with no variable (just a place or year) are skipped, and a variable
    that repeats ("hotter or colder") counts once."""
    seen, out = set(), []
    for clause in CLAUSE_SPLIT.split(q or ""):
        hits = search_question(clause, k=1) if clause.strip() else []
        if not hits or hits[0][0] in seen:
            continue
        seen.add(hits[0][0])
        found = geo.find_places(clause, max_alts=0)
        out.append((hits[0][0], found[0] if found else None))
    return out[:max_vars]


def find_pair(q):
    """A comparison case for a question that names two or more variables, else None. The first two variables
    are compared at the place named next to them (or the question's first place)."""
    vs = question_variables(q)
    if len(vs) < 2:
        return None
    places = geo.find_places(q)
    if not places:
        raise ValueError(no_place_message())
    pa = vs[0][1] or vs[1][1] or places[0]
    pb = vs[1][1] or pa
    ql = q.lower()
    season = season_in(ql)
    if season and pa["lat"] < 0:
        season = FLIP[season]
    yrs = [int(y) for y in re.findall(r"\b(19[89]\d|20[0-2]\d)\b", ql)]
    mk = lambda v, m: dict(variable=v, place=m["label"], lat=m["lat"], lon=m["lon"])
    out = _pair(mk(vs[0][0], pa), mk(vs[1][0], pb), season, min(yrs) if yrs else None,
                max(yrs) if len(yrs) > 1 else None, search_question(q, k=6))
    return {**out, "others": [v for v, _ in vs[2:]]}


def parse_compare_keywords(q):
    pair = find_pair(q)
    if pair:
        return pair
    ql = q.lower()
    hits = search_question(q, k=6)
    places = geo.find_places(q)
    if not hits:
        raise ValueError("I couldn't match that to a variable in the catalog. Try the menus below.")
    if len(places) < 2:
        raise ValueError(no_place_message(2))
    season = season_in(ql)
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
    hits = search_question(q, k=8)
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
    try:
        txt = _llm(system, q, 400)
    except ValueError:
        if USER_API_KEY.get():  # their key failed: still answer, with the keyword parser
            return parse_compare_keywords(q), False
        raise
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
    from planets.analysis import compare as _cmp
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
