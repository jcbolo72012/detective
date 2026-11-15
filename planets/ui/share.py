"""Shareable links: the browser's address always describes what's on screen, and opening it reproduces it.

Readable parameters (so a link in a slide or chat says what it shows):
    ?q=Is Phoenix getting hotter in summer?                      a question (Ask mode)
    ?cq=Is Svalbard warming faster than London?                  a comparison question
    ?v=T2M&at=Phoenix, USA&lat=33.45&lon=-112.07&s=JJA&y=1981-2025      one investigation
    ...&v2=PRECTOTCORR&at2=...&lat2=..&lon2=..                   two (a comparison built from menus)

Everything read from a link goes through nl.validate (unknown variables refused, years and places clamped), so
a hand-edited link can't make the app do anything a user couldn't do from the menus. Tokens and keys are never
put in links. A Giovanni or PO.DAAC variable in a link still needs the visitor's own Earthdata token.
"""
import re
from urllib.parse import urlencode, urlsplit

# A place name from a link is shown in headings, which render Markdown: keep only plain characters so a crafted
# link can't add clickable links, images or formatting to a page that looks like ours.
_PLAIN = re.compile(r"[^\w\s,.'()\-]", re.UNICODE)


def clean_label(text, limit=60):
    t = _PLAIN.sub("", str(text or "")).strip()
    return re.sub(r"\s+", " ", t)[:limit].strip() or "Shared place"

FIELDS = ("variable", "place", "lat", "lon", "season", "start_year", "end_year")


def _side(p, n=""):
    return {f"v{n}": p["variable"], f"at{n}": p["place"], f"lat{n}": f"{round(float(p['lat']), 4):g}",
            f"lon{n}": f"{round(float(p['lon']), 4):g}"}


def case_params(p):
    """One investigation -> link parameters."""
    out = _side(p)
    if p.get("season"):
        out["s"] = p["season"]
    out["y"] = f"{p['start_year']}-{p['end_year']}"
    return out


def pair_params(case):
    """A comparison {a, b} -> link parameters (season and years taken from A; the comparison shares them)."""
    out = {**case_params(case["a"]), **_side(case["b"], "2")}
    return out


def _parse_side(qp, n, validate, season, y0, y1):
    return validate(dict(variable=qp[f"v{n}"], place=clean_label(qp.get(f"at{n}")),
                         lat=float(qp[f"lat{n}"]), lon=float(qp[f"lon{n}"]),
                         season=season, start_year=y0, end_year=y1))


def read(qp, validate):
    """Query parameters -> what to open, or None:
       ("ask", question) | ("compare_q", question) | ("case", params) | ("pair", {a, b}).
    Raises ValueError (with a message for the user) on a broken or invalid link."""
    qp = {k: (v[-1] if isinstance(v, list) else v) for k, v in dict(qp).items()}
    if qp.get("q"):
        return "ask", qp["q"][:300]
    if qp.get("cq"):
        return "compare_q", qp["cq"][:300]
    if not qp.get("v"):
        return None
    try:
        season = qp.get("s") or None
        y0 = y1 = None
        if qp.get("y"):
            a, _, b = qp["y"].partition("-")
            y0, y1 = int(a), int(b or a)
        a = _parse_side(qp, "", validate, season, y0, y1)
        if qp.get("v2"):
            return "pair", {"a": a, "b": _parse_side(qp, "2", validate, season, y0, y1)}
        return "case", a
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"This link couldn't be opened ({e}). Start a new investigation below.")


def full_url(params, base=None):
    """Absolute link for the current page (base from st.context.url when available), else a relative one."""
    q = urlencode(params)
    if base:
        parts = urlsplit(base)
        return f"{parts.scheme}://{parts.netloc}{parts.path}?{q}" if parts.netloc else f"?{q}"
    return f"?{q}"
