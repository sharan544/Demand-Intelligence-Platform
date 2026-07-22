import streamlit as st
import pandas as pd
import json
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import h3

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Demand Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #F8FAFC; }
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 18px 20px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        text-align: center;
    }
    .metric-val {
        font-size: 2rem;
        font-weight: 700;
        color: #0F2D4A;
        line-height: 1.1;
    }
    .metric-lbl {
        font-size: 0.78rem;
        color: #64748B;
        margin-top: 4px;
    }
    .section-header {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0F2D4A;
        margin-bottom: 0.5rem;
        padding-bottom: 4px;
        border-bottom: 2px solid #E2E8F0;
    }
    .insight-box {
        background: #EBF3FB;
        border-left: 4px solid #1A6FA8;
        border-radius: 6px;
        padding: 10px 14px;
        font-size: 0.88rem;
        color: #1A4068;
        margin-top: 8px;
    }
    .gap-box {
        background: #FFF3CD;
        border-left: 4px solid #E67E22;
        border-radius: 6px;
        padding: 10px 14px;
        font-size: 0.88rem;
        color: #7D5A00;
        margin-top: 8px;
    }
    div[data-testid="stSidebar"] {
        background: #0F2D4A;
    }
    div[data-testid="stSidebar"] * {
        color: white !important;
    }
    div[data-testid="stSidebar"] .stSelectbox label,
    div[data-testid="stSidebar"] .stSlider label {
        color: #B0C4D8 !important;
        font-size: 0.82rem;
    }
</style>
""", unsafe_allow_html=True)

# ── LOAD DATA ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    with open("charlotte_demand_scores.geojson") as f:
        scores_geo = json.load(f)
    with open("charlotte_predictive_demand_2yr.geojson") as f:
        pred_geo = json.load(f)

    # Simple spatial naming helper based on latitude/longitude bounds
    def get_zone_name(lat, lng):
        if lat > 35.28:
            return "University City / North"
        elif lat < 35.12:
            return "Ballantyne / South"
        elif -80.88 < lng < -80.80 and 35.20 < lat < 35.25:
            return "Uptown / Central Business District"
        elif lng < -80.88:
            return "Airport / West Metro"
        elif lng > -80.78:
            return "East Charlotte / Plaza Midwood"
        else:
            return "Charlotte Sub-Metro Corridor"

    def geo_to_df(geo):
        rows = []
        for feat in geo["features"]:
            props = feat["properties"].copy()
            coords = feat["geometry"]["coordinates"][0]
            
            # Extract centroid
            lngs = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            lat_center = sum(lats) / len(lats)
            lng_center = sum(lngs) / len(lngs)
            
            props["lat"] = lat_center
            props["lng"] = lng_center
            props["polygon"] = coords
            
            # Assign human-readable location name
            props["location_name"] = get_zone_name(lat_center, lng_center)
            rows.append(props)
        return pd.DataFrame(rows)

    scores_df = geo_to_df(scores_geo)
    pred_df   = geo_to_df(pred_geo)
    return scores_df, pred_df, scores_geo, pred_geo

# Execute load function to define variables globally
scores_df, pred_df, scores_geo, pred_geo = load_data()

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛰️ Demand Intelligence")
    st.markdown("**Charlotte, NC Pilot**")
    st.markdown("---")

    view_mode = st.selectbox(
        "Score Layer",
        ["Current Growth Score", "2-Year Predictive Score", "Infrastructure Velocity", "Nighttime Light Radiance"]
    )

    st.markdown("---")
    min_score = st.slider("Minimum Score Filter", 0, 100, 0, 5)
    top_n     = st.slider("Top N Hexagons to Highlight", 5, 50, 10, 5)

    st.markdown("---")
    show_table   = st.checkbox("Show Data Table", True)
    show_charts  = st.checkbox("Show Charts", True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#7B8FAB;'>
    <b>Data Sources</b><br>
    • NASA VIIRS Black Marble<br>
    • US Census ACS (Block Group)<br>
    • OpenStreetMap Overpass API<br>
    • Uber H3 Resolution 8<br><br>
    <b>Pipeline</b><br>
    H3 Grid → Census Interpolation → OSM POIs → Satellite → ML Score → FastAPI
    </div>
    """, unsafe_allow_html=True)

# ── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:#0F2D4A; border-radius:10px; padding:18px 24px; margin-bottom:18px;'>
  <h2 style='color:white; margin:0; font-size:1.5rem;'>🛰️ Demand Intelligence Platform</h2>
  <p style='color:#B0C4D8; margin:4px 0 0 0; font-size:0.88rem;'>
    Predicting where consumer demand will emerge in Tier 2 & 3 cities — before the market sees it.
    &nbsp;|&nbsp; Charlotte, NC Pilot &nbsp;|&nbsp; Uber H3 Resolution 8
  </p>
</div>
""", unsafe_allow_html=True)

# ── KPI METRICS ──────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5, col6 = st.columns(6)

kpis = [
    (col1, "2,251",  "Hexagons Scored",       "#0F2D4A"),
    (col2, "1,523",  "Commercial POIs",        "#1A6FA8"),
    (col3, f"{scores_df['growth_score'].max():.1f}", "Top Growth Score", "#1B7F4F"),
    (col4, f"{scores_df['growth_score'].mean():.1f}", "Avg Growth Score", "#1A6FA8"),
    (col5, f"{pred_df['predictive_2yr_score'].max():.1f}", "Top 2yr Score", "#1B7F4F"),
    (col6, "260.3",  "Max NTL Radiance",       "#E67E22"),
]
for col, val, lbl, color in kpis:
    with col:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-val' style='color:{color}'>{val}</div>
            <div class='metric-lbl'>{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── SELECT SCORE COLUMN ───────────────────────────────────────────────────────
score_col_map = {
    "Current Growth Score":      ("growth_score",          scores_df, scores_geo),
    "2-Year Predictive Score":   ("predictive_2yr_score",  pred_df,   pred_geo),
    "Infrastructure Velocity":   ("infra_velocity",        pred_df,   pred_geo),
    "Nighttime Light Radiance":  ("night_lights_radiance", scores_df, scores_geo),
}
score_col, active_df, active_geo = score_col_map[view_mode]

# Filter
filtered_df = active_df[active_df["growth_score"] >= min_score].copy()

# Normalize score col for color
vmin = filtered_df[score_col].quantile(0.05)
vmax = filtered_df[score_col].quantile(0.95)
filtered_df["norm"] = ((filtered_df[score_col] - vmin) / (vmax - vmin + 1e-9)).clip(0, 1)

def score_to_color(n):
    # green → yellow → red (reversed: high = green)
    r = int(255 * (1 - n))
    g = int(200 * n + 55)
    b = 60
    return [r, g, b, 160]

filtered_df["fill_color"] = filtered_df["norm"].apply(score_to_color)

# ── MAP + TABLE LAYOUT ────────────────────────────────────────────────────────
map_col, info_col = st.columns([2, 1])

with map_col:
    st.markdown(f"<div class='section-header' style='color:#FFFFFF !important; border-bottom: 2px solid #334155;'>🗺️ {view_mode} — Charlotte MSA</div>", unsafe_allow_html=True)

    # 1. Map Layer: Polygon layer with semi-transparent fill so street map shows through underneath
    layer = pdk.Layer(
        "PolygonLayer",
        data=filtered_df,
        get_polygon="polygon",
        get_fill_color="fill_color",
        get_line_color=[255, 255, 255, 40], # Thin subtle white borders around hexagons
        line_width_min_pixels=0.8,
        pickable=True,
        auto_highlight=True,
        highlight_color=[255, 255, 100, 200],
    )

    # 2. Viewport centered over Charlotte
    view = pdk.ViewState(
        latitude=35.23,
        longitude=-80.84,
        zoom=10.5,
        pitch=35, # Slightly angled 3D view
        bearing=0,
    )

    # 3. Clean Tooltip
    tooltip = {
        "html": """
        <div style='background:#0F2D4A; color:white; padding:10px; border-radius:8px; font-size:12px; border:1px solid #1A6FA8;'>
            <b>Zone:</b> {location_name}<br>
            <b>H3 Index:</b> <span style='color:#B0C4D8;'>{h3_index}</span><br>
            <hr style='margin:4px 0; border-color:#1A6FA8;'>
            <b>Growth Score:</b> {growth_score}<br>
            <b>Population:</b> {population_density_score}<br>
            <b>Income:</b> ${wealth_score}<br>
            <b>POIs:</b> {commercial_density_score}<br>
            <b>NTL Radiance:</b> {night_lights_radiance}
        </div>
        """,
        "style": {"backgroundColor": "transparent", "color": "white"}
    }

    # 4. FREE BASEMAP: Choose Dark or Light open-source CartoDB style
    # For Dark Mode Streamlit: CartoDB Dark Matter shows streets & labels in sleek dark style
    # For Light Mode Streamlit: Use "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
    CARTO_DARK_MAP = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"

    st.pydeck_chart(pdk.Deck(
        layers=[layer],
        initial_view_state=view,
        tooltip=tooltip,
        map_style=CARTO_DARK_MAP,  # 👈 No Mapbox key needed! Free street map tile layer.
    ), height=480)

    st.markdown("""
    <div class='insight-box'>
    💡 <b>How to read this map:</b> Green hexagons = high growth potential. Red = low. 
    Streets, highways, and district boundaries are visible underneath the transparent hex mesh.
    </div>
    """, unsafe_allow_html=True)

with info_col:
    st.markdown(f"<div class='section-header'>🏆 Top {top_n} Opportunity Zones</div>", unsafe_allow_html=True)

    top_df = active_df.nlargest(top_n, score_col)[
        ["location_name", "h3_index", "wealth_score", "population_density_score", "commercial_density_score", score_col]
    ].copy()
    
    # Combine Location Zone Name + Short H3 Index
    top_df["Target Zone"] = top_df["location_name"] + " (" + top_df["h3_index"].str[:8] + ")"
    
    top_df = top_df[["Target Zone", "wealth_score", "population_density_score", "commercial_density_score", score_col]]
    top_df.columns = ["Target Zone", "Income ($)", "Population", "POIs", "Score"]
    
    top_df["Income ($)"] = top_df["Income ($)"].apply(lambda x: f"${x:,.0f}")
    top_df["Score"] = top_df["Score"].round(2)

    st.dataframe(top_df, use_container_width=True, hide_index=True, height=280)

    # Score gauge for top hexagon
    top_score = active_df[score_col].max()
    st.markdown(f"<div class='section-header' style='margin-top:12px;'>📊 Score Distribution</div>", unsafe_allow_html=True)

    fig_hist = px.histogram(
        filtered_df, x=score_col, nbins=30,
        color_discrete_sequence=["#1A6FA8"],
        labels={score_col: view_mode},
    )
    fig_hist.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        height=180,
        paper_bgcolor="white",
        plot_bgcolor="#F8FAFC",
        showlegend=False,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#E2E8F0"),
        bargap=0.05,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown(f"""
    <div class='gap-box'>
    ⚠️ <b>Data Note:</b> NASA VIIRS resolution is 500m/pixel. Zone-level analysis only — individual asset monitoring requires paid high-res imagery (~$500/mo Planet Labs).
    </div>
    """, unsafe_allow_html=True)

# ── CHARTS ROW ───────────────────────────────────────────────────────────────
# ── CHARTS ROW ───────────────────────────────────────────────────────────────
if show_charts:
    st.markdown("---")
    st.markdown("<div class='section-header' style='color:#FFFFFF !important; border-bottom: 2px solid #334155;'>📈 Analytics Deep Dive</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        # Income vs Growth Score scatter
        fig_scatter = px.scatter(
            scores_df.sample(min(500, len(scores_df))),
            x="wealth_score", y="growth_score",
            color="night_lights_radiance",
            color_continuous_scale="Blues",
            labels={"wealth_score": "Median Income ($)", "growth_score": "Growth Score", "night_lights_radiance": "NTL"},
            title="Income vs Growth Score",
            template="plotly_dark",
            opacity=0.8,
        )
        fig_scatter.update_layout(
            height=300, margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor="rgba(15, 23, 42, 0.6)", 
            plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False,
            font=dict(color="#E2E8F0"),
            title_font=dict(size=14, color="#F8FAFC")
        )
        fig_scatter.update_xaxes(showgrid=True, gridcolor="#334155")
        fig_scatter.update_yaxes(showgrid=True, gridcolor="#334155")
        st.plotly_chart(fig_scatter, use_container_width=True)

    with c2:
        # Current vs Predicted score
        fig_compare = px.scatter(
            pred_df.sample(min(500, len(pred_df))),
            x="growth_score", y="predictive_2yr_score",
            color="infra_velocity",
            color_continuous_scale="Greens",
            labels={"growth_score": "Current Score", "predictive_2yr_score": "2yr Predicted Score", "infra_velocity": "Velocity"},
            title="Current vs 2-Year Predicted Score",
            template="plotly_dark",
            opacity=0.8,
        )
        
        # Diagonal reference line
        max_val = max(pred_df["growth_score"].max(), pred_df["predictive_2yr_score"].max())
        fig_compare.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
            line=dict(color="#F97316", dash="dash", width=2))
            
        fig_compare.update_layout(
            height=300, margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor="rgba(15, 23, 42, 0.6)", 
            plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False,
            font=dict(color="#E2E8F0"),
            title_font=dict(size=14, color="#F8FAFC")
        )
        fig_compare.update_xaxes(showgrid=True, gridcolor="#334155")
        fig_compare.update_yaxes(showgrid=True, gridcolor="#334155")
        st.plotly_chart(fig_compare, use_container_width=True)

    with c3:
        # Top 10 by growth score bar
        top10 = scores_df.nlargest(10, "growth_score")[["h3_index", "growth_score", "wealth_score"]].copy()
        top10["label"] = top10["h3_index"].str[:10] + "..."
        fig_bar = px.bar(
            top10, x="growth_score", y="label",
            orientation="h",
            color="growth_score",
            color_continuous_scale=[[0,"#0284C7"],[1,"#10B981"]],
            title="Top 10 Hexagons by Growth Score",
            template="plotly_dark",
            labels={"growth_score": "Score", "label": ""},
        )
        fig_bar.update_layout(
            height=300, margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor="rgba(15, 23, 42, 0.6)", 
            plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False,
            font=dict(color="#E2E8F0"),
            title_font=dict(size=14, color="#F8FAFC"),
            yaxis=dict(autorange="reversed")
        )
        fig_bar.update_xaxes(showgrid=True, gridcolor="#334155")
        st.plotly_chart(fig_bar, use_container_width=True)

# ── DATA TABLE ────────────────────────────────────────────────────────────────
if show_table:
    st.markdown("---")
    st.markdown("<div class='section-header'>📋 Full Data Table</div>", unsafe_allow_html=True)

    display_df = pred_df[[
        "h3_index", "population_density_score", "wealth_score",
        "commercial_density_score", "night_lights_radiance",
        "ntl_2022", "ntl_2024", "infra_velocity",
        "growth_score", "predictive_2yr_score"
    ]].copy()

    display_df.columns = [
        "H3 Index", "Population", "Income ($)", "POI Count",
        "NTL Radiance", "NTL 2022", "NTL 2024", "Infra Velocity",
        "Growth Score", "2yr Predicted Score"
    ]
    display_df["Income ($)"] = display_df["Income ($)"].apply(lambda x: f"${x:,.0f}")
    display_df["Growth Score"] = display_df["Growth Score"].round(2)
    display_df["2yr Predicted Score"] = display_df["2yr Predicted Score"].round(2)
    display_df["Infra Velocity"] = display_df["Infra Velocity"].round(4)
    display_df = display_df.sort_values("Growth Score", ascending=False)

    st.dataframe(display_df, use_container_width=True, hide_index=True, height=350)

# ── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#94A3B8; font-size:0.78rem; padding:8px 0;'>
    Demand Intelligence Platform &nbsp;|&nbsp; Charlotte Pilot &nbsp;|&nbsp;
    Pipeline: Uber H3 + US Census ACS + OpenStreetMap + NASA VIIRS + Random Forest + FastAPI
</div>
""", unsafe_allow_html=True)