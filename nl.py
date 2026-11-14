"""Language layer. The AI only (1) turns a question into parameters and (2) words the computed numbers.
It never calculates. Works with no API key too (keyword parser + template text).

Set ONE of:  ANTHROPIC_API_KEY  or  OPENAI_API_KEY   (optional: LLM_MODEL to override the model name)"""
import json
import os
import re

from catalog import PLACES
from power import LAST_FULL_YEAR
from retrieve import ALL as VARIABLES, search

SEASON_WORDS = {"summer": "JJA", "winter": "DJF", "spring": "MAM", "fall": "SON", "autumn": "SON"}
FLIP = {"JJA": "DJF", "DJF": "JJA", "MAM": "SON", "SON": "MAM"}


def llm_available():
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY"))


def _llm(system, user, max_tokens=500):
    if os.getenv("ANTHROPIC_API_KEY"):
        import anthropic
        msg = anthropic.Anthropic().messages.create(
            model=os.getenv("LLM_MODEL", "claude-haiku-5-5"), max_tokens=max_tokens,
            system=system, messages=[{"role": "user", "content": user}])
        return msg.content[0].text
    from openai import OpenAI
    out = OpenAI().chat.completions.create(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return out.choices[0].message.content


# ------------------------------------------------------------ question -> parameters
def validate(p):
    """Never trust the model: clamp everything to what the catalog and API support."""
    if p.get("variable") not in VARIABLES:
        raise ValueError("I couldn't match that to a variable in the catalog. Try words like temperature, rain, "
                         "soil moisture, humidity, wind or sunlight.")
    lat, lon = float(p["lat"]), float(p["lon"])
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("Couldn't work out where that place is.")
    season = p.get("season") or None
    if season not in (None, "DJF", "MAM", "JJA", "SON"):
        season = None
    start = min(max(int(p.get("start_year") or 1981), 1981), LAST_FULL_YEAR - 10)
    end = min(max(int(p.get("end_year") or LAST_FULL_YEAR), start + 10), LAST_FULL_YEAR)
    return dict(variable=p["variable"], place=str(p.get("place") or f"{lat:.2f}, {lon:.2f}"),
                lat=lat, lon=lon, season=season, start_year=start, end_year=end)


def parse_keywords(q):
    """No-AI fallback: keyword scoring against the catalog."""
    ql = q.lower()
    hits = search(q, k=5)
    var = hits[0][0] if hits else None
    place = next((n for n in PLACES if any(tok in ql for tok in
                  re.findall(r"[a-z\-]{4,}", n.lower()) if tok not in ("south", "africa", "australia"))), None)
    if place is None:
        raise ValueError("Without an AI key I only know these places: " + ", ".join(PLACES))
    lat, lon = PLACES[place]
    season = next((c for w, c in SEASON_WORDS.items() if w in ql), None)
    if season and lat < 0:
        season = FLIP[season]
    yrs = [int(y) for y in re.findall(r"\b(19[89]\d|20[0-2]\d)\b", ql)]
    p = validate(dict(variable=var, place=place, lat=lat, lon=lon, season=season,
                      start_year=min(yrs) if yrs else None, end_year=max(yrs) if len(yrs) > 1 else None))
    return {**p, "candidates": hits}


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
    return {**p, "candidates": hits}, True


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
