"""Curated variable + place catalog. Edit freely: this is the 'plain-language definitions' deliverable.
`power` = NASA POWER parameter code (live, any point on Earth, 1981-present, no login).
Giovanni CSVs dropped into ./data show up in the app as extra case files."""

VARIABLES = {
    "T2M": dict(name="Air temperature", units="°C", source="NASA POWER (MERRA-2)",
                definition="Average air temperature 2 m above the ground: what a weather station thermometer reads.",
                keywords=["temperature", "hot", "hotter", "warm", "warming", "heat", "cold", "cooler"]),
    "T2M_MAX": dict(name="Daily maximum temperature", units="°C", source="NASA POWER (MERRA-2)",
                    definition="Average of each day's highest temperature. Good for heatwave questions.",
                    keywords=["heatwave", "heat wave", "hottest", "maximum temperature", "daytime high", "extreme heat"]),
    "T2M_MIN": dict(name="Daily minimum temperature", units="°C", source="NASA POWER (MERRA-2)",
                    definition="Average of each day's lowest temperature, usually at night.",
                    keywords=["night", "nights", "minimum temperature", "frost", "coldest", "overnight"]),
    "PRECTOTCORR": dict(name="Precipitation", units="mm/day", source="NASA POWER (MERRA-2, bias-corrected)",
                        definition="Rain plus melted snow, as an average depth of water per day.",
                        keywords=["rain", "rainfall", "precipitation", "wetter", "drier", "drought", "dry", "wet", "monsoon", "snow"]),
    "GWETROOT": dict(name="Root-zone soil wetness", units="fraction (0-1)", source="NASA POWER (MERRA-2)",
                     definition="How wet the top metre of soil is, where plant roots live. 0 = bone dry, 1 = saturated.",
                     keywords=["soil", "soil moisture", "drought", "crops", "farming", "agriculture", "drying out"]),
    "RH2M": dict(name="Relative humidity", units="%", source="NASA POWER (MERRA-2)",
                 definition="How much water vapour the air holds compared with the most it could hold.",
                 keywords=["humidity", "humid", "muggy", "dry air"]),
    "WS2M": dict(name="Wind speed", units="m/s", source="NASA POWER (MERRA-2)",
                 definition="Average wind speed 2 m above the ground.",
                 keywords=["wind", "windy", "windier", "breeze", "stilling"]),
    "ALLSKY_SFC_SW_DWN": dict(name="Sunlight reaching the surface", units="kWh/m²/day", source="NASA POWER (CERES)",
                              definition="Solar energy arriving at the ground, clouds included. Drops when skies get cloudier or hazier.",
                              keywords=["sun", "sunlight", "sunny", "solar", "cloudy", "cloudier", "dimming", "brightening"]),
}

# Follow-up variables worth checking next (used when no AI key is set). Reasons are hypotheses, not causes.
RELATED = {
    "T2M": [("PRECTOTCORR", "to see whether rainfall is changing alongside temperature at this place."),
            ("T2M_MIN", "to see whether nights are changing faster or slower than the daily average.")],
    "T2M_MAX": [("T2M_MIN", "to see whether nights are changing as much as the daytime highs."),
                ("GWETROOT", "to see whether the soil is drying, since dry ground often goes with hotter days.")],
    "T2M_MIN": [("RH2M", "to see whether humidity is changing, since moist air tends to keep nights warmer."),
                ("T2M_MAX", "to see whether daytime highs are changing as much as the nights.")],
    "PRECTOTCORR": [("GWETROOT", "to see whether the change in rain shows up in the soil where plants grow."),
                    ("T2M", "to see whether temperature is changing too, which affects how fast water evaporates.")],
    "GWETROOT": [("PRECTOTCORR", "to see whether rainfall is changing in the same direction as the soil."),
                 ("T2M", "to see whether it is getting warmer, which can dry soil even when rain holds steady.")],
    "RH2M": [("T2M", "to see whether temperature is changing, since warmer air can hold more water vapour."),
             ("PRECTOTCORR", "to see whether rainfall is changing along with the humidity.")],
    "WS2M": [("T2M", "to see whether temperature is changing at the same time as the wind."),
             ("ALLSKY_SFC_SW_DWN", "to see whether sunlight at the surface is changing, a sign of shifting weather patterns.")],
    "ALLSKY_SFC_SW_DWN": [("PRECTOTCORR", "to see whether rainfall is changing, since cloudier skies usually bring both."),
                          ("T2M", "to see whether temperature is following the change in sunlight.")],
}

PLACES = {  # name: (lat, lon)
    "Phoenix, USA": (33.45, -112.07), "New York, USA": (40.71, -74.01), "Miami, USA": (25.76, -80.19),
    "Fairbanks, Alaska": (64.84, -147.72), "Central Valley, California": (36.6, -119.9),
    "Sahel (Niamey, Niger)": (13.51, 2.11), "Cape Town, South Africa": (-33.92, 18.42),
    "Delhi, India": (28.61, 77.21), "Beijing, China": (39.90, 116.41), "London, UK": (51.51, -0.13),
    "Amazon (Manaus, Brazil)": (-3.12, -60.02), "Sydney, Australia": (-33.87, 151.21),
    "Svalbard, Arctic": (78.22, 15.65), "Murray-Darling Basin, Australia": (-33.5, 146.0),
}

DEMO_QUESTIONS = [
    "Is Phoenix getting hotter in summer?",
    "Is the Sahel getting wetter or drier?",
    "Are nights getting warmer in Delhi?",
    "Is soil drying out in California's Central Valley?",
    "Is Svalbard in the Arctic getting warmer?",
]
