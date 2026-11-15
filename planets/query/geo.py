"""Offline place lookup: question text -> coordinates, with no API key and no network.

Order of preference for each name found in a question:
  preset places (catalog.PLACES, incl. regions like the Sahel) > countries > US states > cities.
Countries and states are represented by their largest city, and the label says so.
City list: GeoNames via the `geonamescache` package (cities over 15,000 people, CC BY 4.0).
Without that package installed, only the preset places work."""
import re
import unicodedata

from catalog import PLACES

try:
    import geonamescache
    AVAILABLE = True
except ImportError:
    AVAILABLE = False

ATTRIBUTION = "Place names: GeoNames (geonames.org, CC BY 4.0) via geonamescache"

# extra names for presets (keys are folded: lowercase, no accents, hyphens as spaces)
PRESET_ALIASES = {
    "arctic": "Svalbard, Arctic", "niamey": "Sahel (Niamey, Niger)", "manaus": "Amazon (Manaus, Brazil)",
    "amazon rainforest": "Amazon (Manaus, Brazil)", "murray darling": "Murray-Darling Basin, Australia",
    "new delhi": "Delhi, India", "nyc": "New York, USA", "new york city": "New York, USA",
    "central valley": "Central Valley, California",
}
COUNTRY_ALIASES = {"usa": "US", "united states of america": "US", "america": "US",
                   "uk": "GB", "britain": "GB", "great britain": "GB", "england": "GB", "scotland": "GB",
                   "wales": "GB", "holland": "NL", "the netherlands": "NL", "uae": "AE", "korea": "KR",
                   "ivory coast": "CI", "czechia": "CZ", "burma": "MM"}
CITY_ALIASES = {"washington dc": ("Washington", "US"), "dc": ("Washington", "US")}
SHORT_OK = {"uk", "dc", "usa", "uae"}  # names under 3 letters (or stop-like) that may still match

# words that are also names of real towns but almost never mean a place in a question
STOP = set("""a an the is are was were be been being am it its it's this that these those there here of in on at to
for from by with and or but not no nor than then while whereas as so if when where why how what which who whom whose
do does did done doing has have had having will would can could should may might must shall get gets getting got
gotten become becoming became more most less least much many very over under above below between both same different
other another each every any all some such only just still also even ever really up down out off into onto about
again before after since until during year years month months day days night nights week season seasons time times
summer winter spring fall autumn trend trends change changes changing changed rate faster slower quicker hotter
colder warmer cooler wetter drier windier sunnier cloudier muggier humid hot cold warm cool wet dry dryer windy sunny
cloudy rain rains rainy rainfall snow snowy wind winds sun sunlight cloud clouds heat heatwave heatwaves drought
droughts frost soil moisture humidity temperature temperatures pressure air sea ocean coast land water ice weather
climate global world earth local area areas region regions place places city cities town country countries north
south east west northern southern eastern western central new old big small high low best worst good bad compare
compared comparison versus vs like near around average mean data nasa show tell me my our we you i decade decades
century percent degrees plus minus long short lake river mountain mountains valley desert forest island islands
beach bay port""".split())


def fold(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"'s\b", "", s)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.replace("-", " "))).strip()


def _preset_key(name):
    return fold(re.split(r"[,(]", name)[0])


_INDEX = None


def _index():
    """Built once per process: folded name -> list of candidate places (best first)."""
    global _INDEX
    if _INDEX is not None:
        return _INDEX
    idx = {}

    def add(key, entry):
        if key:
            idx.setdefault(key, []).append(entry)

    for name, (lat, lon) in PLACES.items():
        add(_preset_key(name), dict(label=name, lat=lat, lon=lon, kind="preset", rank=0))
    for alias, name in PRESET_ALIASES.items():
        lat, lon = PLACES[name]
        add(alias, dict(label=name, lat=lat, lon=lon, kind="preset", rank=0))
    if AVAILABLE:
        gc = geonamescache.GeonamesCache(min_city_population=15000)
        cities = sorted(gc.get_cities().values(), key=lambda c: -c["population"])
        countries = gc.get_countries()
        states = gc.get_us_states()
        cname = {code: c["name"] for code, c in countries.items()}
        sname = {code: s["name"] for code, s in states.items()}
        largest_in_country, largest_in_state = {}, {}
        for c in cities:  # population order, so the first seen is the largest
            largest_in_country.setdefault(c["countrycode"], c)
            if c["countrycode"] == "US":
                largest_in_state.setdefault(c["admin1code"], c)

        def city_label(c):
            if c["countrycode"] == "US":
                return f"{c['name']}, {c['admin1code']}, USA"
            return f"{c['name']}, {cname.get(c['countrycode'], c['countrycode'])}"

        for code, c in countries.items():
            big = largest_in_country.get(code)
            if big:
                e = dict(label=f"{c['name']} (at its largest city, {big['name']})", lat=big["latitude"],
                         lon=big["longitude"], kind="country", rank=1, country=code)
                add(fold(c["name"]), e)
        for alias, code in COUNTRY_ALIASES.items():
            if alias == fold(cname.get(code, "")):
                continue
            for e in list(idx.get(fold(cname.get(code, "")), [])):
                if e["kind"] == "country":
                    add(alias, e)
        for code, s in states.items():
            big = largest_in_state.get(code)
            if big:
                add(fold(s["name"]), dict(label=f"{s['name']}, USA (at its largest city, {big['name']})",
                                          lat=big["latitude"], lon=big["longitude"], kind="state", rank=2,
                                          state=code, country="US"))
        seen_alt = set()
        for c in cities:
            e = dict(label=city_label(c), lat=c["latitude"], lon=c["longitude"], kind="city", rank=3,
                     country=c["countrycode"], state=c["admin1code"] if c["countrycode"] == "US" else None,
                     pop=c["population"])
            add(fold(c["name"]), e)
            if c["population"] >= 300000:  # well-known old/alternate names: Bombay, Peking, Saigon...
                for alt in c.get("alternatenames", []):
                    k = fold(alt)
                    if (alt.isascii() and len(k) >= 4 and alt[:1].isupper() and k not in STOP
                            and k != fold(c["name"]) and (k, c["geonameid"]) not in seen_alt):
                        seen_alt.add((k, c["geonameid"]))
                        add(k, {**e, "rank": 4})
        for alias, (name, cc) in CITY_ALIASES.items():
            for e in list(idx.get(fold(name), [])):
                if e["kind"] == "city" and e["country"] == cc:
                    add(alias, e)
                    break
    for k, v in idx.items():  # best first, no duplicate labels
        out, seen = [], set()
        for e in sorted(v, key=lambda e: (e["rank"], -e.get("pop", 0))):
            if e["label"] not in seen:
                seen.add(e["label"])
                out.append(e)
        idx[k] = out
    _INDEX = idx
    return idx


def _public(e):
    return dict(label=e["label"], lat=round(float(e["lat"]), 4), lon=round(float(e["lon"]), 4), kind=e["kind"])


def find_places(text, allow_stop=False, max_alts=3):
    """All places named in `text`, in the order they appear:
    [{text, label, lat, lon, kind, alts: [{label, lat, lon, kind}, ...]}]"""
    idx = _index()
    words = fold(text).split()
    used = [False] * len(words)
    found = []  # (start, end, key, entries)
    for n in (4, 3, 2, 1):
        for i in range(len(words) - n + 1):
            if any(used[i:i + n]):
                continue
            key = " ".join(words[i:i + n])
            if n == 1 and not allow_stop and key not in SHORT_OK and (key in STOP or len(key) < 3 or key.isdigit()):
                continue
            if key in idx:
                found.append((i, i + n, key, idx[key]))
                for j in range(i, i + n):
                    used[j] = True
    found.sort()
    # "Paris, Texas" / "Portland Maine" / "Valencia, Venezuela": a place followed by a state or country narrows it
    merged, skip = [], set()
    for a, (s, e, key, ents) in enumerate(found):
        if a in skip:
            continue
        # "California's Central Valley" / "Texas' Houston": a state or country directly before a more specific place
        if ents[0]["kind"] in ("state", "country") and a + 1 < len(found) and found[a + 1][0] - e <= 1 \
                and found[a + 1][3][0]["kind"] in ("preset", "city"):
            region = ents[0]
            nxt = found[a + 1][3]
            field = "state" if region["kind"] == "state" else "country"
            inside = [x for x in nxt if x["kind"] == "city" and x.get(field) == region[field]]
            if inside:
                found[a + 1] = (*found[a + 1][:3], inside + [x for x in nxt if x not in inside])
            continue
        if a + 1 < len(found):
            s2, e2, key2, ents2 = found[a + 1]
            region = next((x for x in ents2 if x["kind"] in ("state", "country")), None)
            if region and s2 - e <= 1 and len(ents) > 1 or (region and s2 - e <= 1 and ents[0]["kind"] == "city"):
                if region["kind"] == "state":
                    narrowed = [x for x in ents if x["kind"] == "city" and x.get("state") == region["state"]]
                else:
                    narrowed = [x for x in ents if x["kind"] == "city" and x.get("country") == region["country"]]
                if narrowed:
                    ents = narrowed + [x for x in ents if x not in narrowed]
                    skip.add(a + 1)
        best = ents[0]
        # alternatives: genuinely different places only (not the same city listed twice)
        alts = [x for x in ents[1:] if abs(x["lat"] - best["lat"]) > 0.5 or abs(x["lon"] - best["lon"]) > 0.5]
        merged.append(dict(text=key, **_public(best), alts=[_public(x) for x in alts[:max_alts]]))
    out, labels = [], set()
    for m in merged:
        if m["label"] not in labels:
            labels.add(m["label"])
            out.append(m)
    return out


def lookup(text):
    """Best single place for a typed place name (stop-words allowed), or None."""
    hits = find_places(text, allow_stop=True)
    return hits[0] if hits else None
