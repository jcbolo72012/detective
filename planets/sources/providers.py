"""Which NASA service owns a catalog id, and what the user must supply to use it.

Catalog ids: NASA POWER codes as-is (T2M); everything else carries a prefix so ids never clash.
Giovanni and PO.DAAC use the Earthdata token the user already pastes in the sidebar."""

PREFIXES = {"GIO:": "giovanni", "POD:": "podaac"}
NEEDS_EARTHDATA = frozenset({"giovanni", "podaac"})
LABELS = {"power": "NASA POWER", "giovanni": "NASA Giovanni", "podaac": "NASA PO.DAAC"}


def provider_of(code):
    code = str(code)
    return next((p for pre, p in PREFIXES.items() if code.startswith(pre)), "power")


def label_of(code):
    return LABELS[provider_of(code)]


def needs_token(code):
    return provider_of(code) in NEEDS_EARTHDATA


def strip_prefix(code):
    """'POD:SST' -> 'SST'."""
    code = str(code)
    return next((code[len(pre):] for pre in PREFIXES if code.startswith(pre)), code)


def external_catalogs():
    """Catalog entries of the sources that are not POWER and not Giovanni (which has its own loader)."""
    try:
        import podaac
    except ImportError:  # netCDF4 missing: the ocean variables just don't appear
        return {}
    return podaac.load()
