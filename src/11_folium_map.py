"""
Stage 11 — Folium Interactive Map
-----------------------------------
Since the Mapillary Vistas images have no embedded GPS, this script
assigns approximate city-level coordinates by hashing the image stem
to one of 12 real world urban locations (London, Paris, NYC, Berlin,
Tokyo, Sydney, Toronto, Amsterdam, Barcelona, Singapore, Chicago, Dubai).
This simulates a geotagged dataset for map demonstration purposes —
the assignment is deterministic (hash-based) so results are reproducible.

Outputs:
  outputs/geoscenefm_map.html   — Folium choropleth marker map
  outputs/logs/11_folium_map.log
"""
import os, sys, json, hashlib, logging
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster, MiniMap, Fullscreen

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

LOG_PATH = os.path.join(config.LOG_DIR, "11_folium_map.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── 12 real urban centres used as simulated GPS locations ─────────────────────
URBAN_CENTRES = [
    {"city": "London",      "lat":  51.5074, "lon":  -0.1278, "country": "UK"},
    {"city": "Paris",       "lat":  48.8566, "lon":   2.3522, "country": "France"},
    {"city": "New York",    "lat":  40.7128, "lon": -74.0060, "country": "USA"},
    {"city": "Berlin",      "lat":  52.5200, "lon":  13.4050, "country": "Germany"},
    {"city": "Tokyo",       "lat":  35.6762, "lon": 139.6503, "country": "Japan"},
    {"city": "Sydney",      "lat": -33.8688, "lon": 151.2093, "country": "Australia"},
    {"city": "Toronto",     "lat":  43.6532, "lon": -79.3832, "country": "Canada"},
    {"city": "Amsterdam",   "lat":  52.3676, "lon":   4.9041, "country": "Netherlands"},
    {"city": "Barcelona",   "lat":  41.3851, "lon":   2.1734, "country": "Spain"},
    {"city": "Singapore",   "lat":   1.3521, "lon": 103.8198, "country": "Singapore"},
    {"city": "Chicago",     "lat":  41.8781, "lon": -87.6298, "country": "USA"},
    {"city": "Dubai",       "lat":  25.2048, "lon":  55.2708, "country": "UAE"},
]

# Cluster → hex colour (matches Stage 4 palette)
CLUSTER_COLORS = [
    "#E74C3C", "#2980B9", "#27AE60", "#8E44AD",
    "#F39C12", "#16A085", "#D35400", "#7F8C8D",
]

SCENE_GROUP_COLORS = {
    "road_scene" : "#E74C3C",
    "sky_scene"  : "#5DADE2",
    "built_env"  : "#F39C12",
    "green_space": "#27AE60",
    "unknown"    : "#95A5A6",
    "mixed"      : "#8E44AD",
}


def city_for_image(image_id: str) -> dict:
    """Deterministically assign a city based on the image_id hash."""
    h = int(hashlib.md5(image_id.encode()).hexdigest(), 16)
    return URBAN_CENTRES[h % len(URBAN_CENTRES)]


def jitter(lat: float, lon: float, radius: float = 0.04) -> tuple[float, float]:
    """Add small reproducible random offset so markers don't overlap exactly."""
    rng = np.random.default_rng(abs(hash((lat, lon, radius))) % (2**32))
    return lat + rng.uniform(-radius, radius), lon + rng.uniform(-radius, radius)


def make_marker(row: pd.Series) -> folium.CircleMarker:
    cluster_id  = int(row.get("cluster_id", 0))
    scene_group = str(row.get("scene_group", "unknown"))
    image_id    = str(row["image_id"])
    city        = str(row.get("city", "Unknown"))
    country     = str(row.get("country", ""))
    dominant    = str(row.get("dominant_category", "—"))
    top3        = str(row.get("top3_categories", "—"))

    lat = float(row["lat"])
    lon = float(row["lon"])
    color = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]

    popup_html = f"""
    <div style="font-family:sans-serif; font-size:12px; width:220px">
      <b>{image_id}</b><br>
      <hr style="margin:4px 0">
      <b>City:</b> {city}, {country}<br>
      <b>Cluster:</b> {cluster_id}<br>
      <b>Scene Group:</b> {scene_group}<br>
      <b>Dominant Cat:</b> {dominant}<br>
      <b>Top 3:</b> {top3}<br>
    </div>"""

    return folium.CircleMarker(
        location=[lat, lon],
        radius=7,
        color="white",
        weight=0.8,
        fill=True,
        fill_color=color,
        fill_opacity=0.82,
        popup=folium.Popup(popup_html, max_width=240),
        tooltip=f"{image_id} | {scene_group} | C{cluster_id}",
    )


def main():
    log.info("=" * 60)
    log.info("STAGE 11 — Folium Interactive Map")
    log.info("=" * 60)

    meta_path   = os.path.join(config.DATA_DIR, "metadata.csv")
    assign_path = os.path.join(config.STATISTICS_DIR, "cluster_assignments.csv")

    for p in [meta_path, assign_path]:
        if not os.path.exists(p):
            log.error(f"Required file missing: {p}")
            sys.exit(1)

    df_meta   = pd.read_csv(meta_path)
    df_assign = pd.read_csv(assign_path)

    df = df_assign.merge(
        df_meta[["image_id", "scene_group", "dominant_category", "top3_categories"]],
        on="image_id", how="left"
    )
    df["scene_group"]      = df["scene_group"].fillna("unknown")
    df["dominant_category"]= df["dominant_category"].fillna("—")
    df["top3_categories"]  = df["top3_categories"].fillna("—")

    # Assign deterministic cities + jittered coordinates
    cities, lats, lons, countries = [], [], [], []
    for _, row in df.iterrows():
        c = city_for_image(row["image_id"])
        lat, lon = jitter(c["lat"], c["lon"], radius=0.06)
        cities.append(c["city"])
        lats.append(round(lat, 6))
        lons.append(round(lon, 6))
        countries.append(c["country"])

    df["city"]    = cities
    df["lat"]     = lats
    df["lon"]     = lons
    df["country"] = countries

    log.info(f"Assigned {len(df)} images to {len(URBAN_CENTRES)} urban centres.")
    log.info("City distribution:")
    for city, cnt in pd.Series(cities).value_counts().items():
        log.info(f"  {city:15s}: {cnt}")

    # ── Build Folium map ──────────────────────────────────────────────────────
    m = folium.Map(
        location=[30.0, 15.0],
        zoom_start=2,
        tiles="CartoDB positron",
        attr="© OpenStreetMap contributors © CartoDB",
    )

    # City centre markers
    for centre in URBAN_CENTRES:
        folium.Marker(
            location=[centre["lat"], centre["lon"]],
            icon=folium.Icon(color="darkblue", icon="city", prefix="fa"),
            tooltip=f"{centre['city']}, {centre['country']}",
        ).add_to(m)

    # One MarkerCluster layer per scene_group
    scene_groups = sorted(df["scene_group"].unique())
    for sg in scene_groups:
        sg_df = df[df["scene_group"] == sg]
        color = SCENE_GROUP_COLORS.get(sg, "#AAAAAA")
        layer = MarkerCluster(
            name=f"Scene: {sg} ({len(sg_df)})",
            options={
                "maxClusterRadius": 40,
                "iconCreateFunction": f"""
                    function(cluster) {{
                        return L.divIcon({{
                            html: '<div style="background:{color};color:white;'
                                + 'border-radius:50%;width:34px;height:34px;'
                                + 'display:flex;align-items:center;justify-content:center;'
                                + 'font-weight:bold;font-size:12px;border:2px solid white">'
                                + cluster.getChildCount() + '</div>',
                            className: '', iconSize: [34,34]
                        }});
                    }}
                """,
            },
        )
        for _, row in sg_df.iterrows():
            make_marker(row).add_to(layer)
        layer.add_to(m)

    # Layer control + plugins
    folium.LayerControl(collapsed=False).add_to(m)
    MiniMap(toggle_display=True, zoom_level_offset=-6).add_to(m)
    Fullscreen().add_to(m)

    # Legend
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:white;padding:14px 18px;border-radius:8px;
                border:1px solid #ccc;font-family:sans-serif;font-size:12px;
                box-shadow:2px 2px 6px rgba(0,0,0,.15)">
      <b style="font-size:13px">GeoSceneFM</b><br>
      <i>Cluster Colour (KMeans)</i><br>
    """
    for i, col in enumerate(CLUSTER_COLORS):
        legend_html += (
            f'<span style="display:inline-block;width:14px;height:14px;'
            f'border-radius:50%;background:{col};margin-right:6px;'
            f'vertical-align:middle"></span>Cluster {i}<br>'
        )
    legend_html += "<br><i>Marker layers = Scene Groups</i></div>"
    m.get_root().html.add_child(folium.Element(legend_html))

    # Title
    title_html = """
    <div style="position:fixed;top:12px;left:50%;transform:translateX(-50%);
                z-index:1000;background:rgba(255,255,255,.92);
                padding:8px 20px;border-radius:6px;border:1px solid #ddd;
                font-family:sans-serif;font-size:14px;font-weight:bold;
                box-shadow:1px 1px 4px rgba(0,0,0,.15)">
      GeoSceneFM — Urban Scene Embedding Map (Simulated GPS)
    </div>"""
    m.get_root().html.add_child(folium.Element(title_html))

    out_path = os.path.join(config.OUTPUT_DIR, "geoscenefm_map.html")
    m.save(out_path)
    log.info(f"geoscenefm_map.html saved: {out_path}")

    # Save coordinate table
    coord_path = os.path.join(config.STATISTICS_DIR, "image_coordinates.csv")
    df[["image_id", "city", "country", "lat", "lon", "cluster_id", "scene_group"]].to_csv(
        coord_path, index=False
    )
    log.info(f"image_coordinates.csv saved: {coord_path}")

    log.info("=" * 60)
    log.info("STAGE 11 COMPLETE")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
