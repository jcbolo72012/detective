"""The app's "Earth seen from space" look: a starfield, a faint Earth-glow along the bottom, aurora-green accents.
Base colours come from .streamlit/config.toml; this adds what the theme settings can't."""
import streamlit as st

from planets.ui.palette import AURORA, SPACE

# Stars: three tiled layers of tiny dots at different spacings, so the pattern doesn't look like a grid.
_STARS = ", ".join(
    f"radial-gradient(circle at {x}px {y}px, rgba(255,255,255,{a}) 0, rgba(255,255,255,{a}) {r}px, transparent {r + .6}px)"
    for x, y, a, r in ((23, 41, .6, 1.1), (97, 13, .35, .8), (151, 87, .55, 1.2), (61, 133, .3, .8),
                       (181, 167, .5, 1.0), (127, 199, .28, .8)))

CSS = f"""
<style>
.stApp {{
  background:
    radial-gradient(ellipse 120% 45% at 50% 112%, rgba(61,220,151,.38), rgba(79,163,255,.28) 40%, rgba(79,163,255,.06) 62%, transparent 72%),
    {_STARS},
    radial-gradient(ellipse at 20% 0%, #0d2a55 0%, {SPACE} 60%);
  background-size: 100% 100%, 211px 223px, 211px 223px, 211px 223px, 211px 223px, 211px 223px, 211px 223px, 100% 100%;
  background-attachment: fixed;
}}
[data-testid="stSidebar"] > div:first-child {{ background: rgba(6, 20, 46, .82); backdrop-filter: blur(4px); }}
[data-testid="stHeader"] {{ background: transparent; }}
h1 {{ color: {AURORA} !important; letter-spacing: .5px; text-shadow: 0 0 18px rgba(61,220,151,.35); }}
h3, h4 {{ color: #bfe9d6; }}
[data-testid="stMetric"], [data-testid="stExpander"] details {{
  background: rgba(11, 33, 66, .55); border: 1px solid rgba(79,163,255,.18); border-radius: 12px;
}}
[data-testid="stMetric"] {{ padding: .4rem .7rem; }}
[data-testid="stImage"] img, [data-testid="stPyplot"] img {{ border-radius: 10px; }}
</style>
"""


def apply():
    st.markdown(CSS, unsafe_allow_html=True)
