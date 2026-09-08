"""
AquaScan — AI Waterway Pollution Reporter
Frontend & Geospatial Mapping Interface (Person B)

Day 2 additions:
    • Image upload saved to data/uploads/
    • GPS coordinate inputs with one-click waterway presets
    • Mock classification pipeline (swap with model/predict.py on Day 7)
    • SQLite report logging via app/db.py
    • Live report counter in sidebar
    • Folium map renders all logged reports
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import streamlit as st
from PIL import Image

# ── Make sure imports from sibling directories work when running from app/ ──
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Dynamically import db and mock — robust to running from project root or app/
import importlib.util as _ilu

_db_spec = _ilu.spec_from_file_location("db", ROOT / "app" / "db.py")
_db = _ilu.module_from_spec(_db_spec)
_db_spec.loader.exec_module(_db)
init_db = _db.init_db
add_report = _db.add_report
get_reports_df = _db.get_reports_df
get_report_count = _db.get_report_count

_mock_spec = _ilu.spec_from_file_location("mock", ROOT / "model" / "mock.py")
_mock = _ilu.module_from_spec(_mock_spec)
_mock_spec.loader.exec_module(_mock)
predict_image = _mock.mock_predict   # ← swap to model/predict.py on Day 7

import folium
from streamlit_folium import st_folium

# ────────────────────────────────────────────────────────────────────────────
# Page Config (must be the very first Streamlit call)
# ────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AquaScan — Waterway Pollution Reporter",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Initialise database on every startup (safe — uses CREATE IF NOT EXISTS) ─
init_db()

# ── Upload storage directory ─────────────────────────────────────────────────
UPLOAD_DIR = ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── Severity badge colours ───────────────────────────────────────────────────
SEVERITY_COLOURS = {
    "Critical": "#f85149",
    "High":     "#d29922",
    "Medium":   "#388bfd",
    "Low":      "#3fb950",
}

# ── GPS presets for Indian waterway locations ────────────────────────────────
GPS_PRESETS = {
    "— choose a preset —":          (None, None),
    "🏞️  Yamuna Riverbank, Delhi":    (28.6139, 77.2090),
    "🌊  Mithi River Outfall, Mumbai": (19.0760, 72.8777),
    "🕌  Ganges Ghat, Varanasi":      (25.3176, 83.0062),
    "🏖️  Juhu Beach, Mumbai":          (19.0990, 72.8265),
    "🌿  Sabarmati River, Ahmedabad": (23.0225, 72.5714),
    "🐟  Chilika Lake, Odisha":       (19.7147, 85.3220),
    "🌴  Backwaters, Kochi":          (9.9312,  76.2673),
}

# ── Map marker colours per severity ─────────────────────────────────────────
MARKER_COLOURS = {
    "Critical": "red",
    "High":     "orange",
    "Medium":   "blue",
    "Low":      "green",
}


# ────────────────────────────────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ System Status")
    st.success("✅  Frontend: Streamlit Active")
    st.info("🗄️  Storage: SQLite (Local)")

    report_count = get_report_count()
    st.metric(label="📋 Total Reports Logged", value=report_count)

    st.write("---")
    st.markdown("### 📌 Quick Guide")
    st.markdown(
        """
        1. **Scan & Report:** Upload a photo of waterway debris.
        2. **Set Location:** Enter GPS coords or pick a preset.
        3. **Submit:** Save the report to the database.
        4. **Explore Hotspots:** View logged reports on the map.
        """
    )
    st.write("---")
    st.caption("🤖 Classifier: Mock stub (Day 2) — EfficientNetB0 arrives Day 7")


# ────────────────────────────────────────────────────────────────────────────
# Tab Navigation
# ────────────────────────────────────────────────────────────────────────────
tab_scan, tab_map, tab_metrics = st.tabs(
    ["📸 Scan & Report", "🗺️ Pollution Map", "📊 Model Performance & About"]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Scan & Report
# ══════════════════════════════════════════════════════════════════════════════
with tab_scan:
    st.subheader("📸 Report Aquatic Debris")
    st.caption(
        "Upload a photo of waste found near a river, lake, or coastline. "
        "The AI will classify it and log the report to the pollution database."
    )

    # ── Image Upload ─────────────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "Choose an image of waste found near a waterway…",
        type=["jpg", "jpeg", "png"],
        key="uploader",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        col_img, col_result = st.columns([1, 1], gap="large")

        with col_img:
            st.image(image, caption="📷 Uploaded Image", use_container_width=True)

        with col_result:
            st.markdown("#### 🤖 AI Classification Result")

            # Run mock (or real on Day 7) classifier
            predicted_class, confidence, severity = predict_image(image)

            # Display prediction card
            sev_colour = SEVERITY_COLOURS.get(severity, "#888")
            st.markdown(
                f"""
                <div style="
                    background: rgba(255,255,255,0.04);
                    border: 1px solid #30363d;
                    border-left: 5px solid {sev_colour};
                    border-radius: 10px;
                    padding: 18px 20px;
                    margin-bottom: 14px;
                ">
                    <div style="font-size:1.5rem; font-weight:700; color:#f0f6fc;">
                        {predicted_class.upper()}
                    </div>
                    <div style="color:#8b949e; margin-top:4px;">
                        Confidence: <strong style="color:#f0f6fc;">{confidence:.0%}</strong>
                        &nbsp;|&nbsp;
                        Severity:
                        <span style="
                            background:{sev_colour};
                            color:#fff;
                            padding:2px 10px;
                            border-radius:12px;
                            font-size:0.85rem;
                            font-weight:600;
                        ">{severity}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── GPS Location Section ─────────────────────────────────────────────
        st.divider()
        st.markdown("#### 📍 Report Location")

        preset_choice = st.selectbox(
            "Quick Preset Locations (or enter custom coordinates below)",
            options=list(GPS_PRESETS.keys()),
            index=0,
            key="preset",
        )

        preset_lat, preset_lon = GPS_PRESETS[preset_choice]

        col_lat, col_lon = st.columns(2)
        with col_lat:
            lat = st.number_input(
                "Latitude",
                min_value=-90.0,
                max_value=90.0,
                value=preset_lat if preset_lat is not None else 20.5937,
                format="%.4f",
                key="lat_input",
            )
        with col_lon:
            lon = st.number_input(
                "Longitude",
                min_value=-180.0,
                max_value=180.0,
                value=preset_lon if preset_lon is not None else 78.9629,
                format="%.4f",
                key="lon_input",
            )

        # ── Submit Button ────────────────────────────────────────────────────
        st.divider()
        submit_col, _ = st.columns([1, 3])
        with submit_col:
            submitted = st.button(
                "✅  Save Report to Database",
                type="primary",
                use_container_width=True,
            )

        if submitted:
            # Save the uploaded image to disk with a timestamped filename
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = Path(uploaded_file.name).suffix or ".jpg"
            save_filename = f"{timestamp_str}_{predicted_class}{ext}"
            save_path = UPLOAD_DIR / save_filename

            image.save(str(save_path))

            # Write report to SQLite
            new_id = add_report(
                image_path=str(save_path),
                predicted_class=predicted_class,
                confidence=confidence,
                severity=severity,
                lat=lat,
                lon=lon,
            )

            st.success(
                f"✅  Report #{new_id} logged successfully! "
                f"**{predicted_class.capitalize()}** detected at "
                f"({lat:.4f}, {lon:.4f})."
            )
            st.toast(f"🗄️ Report #{new_id} saved to database!", icon="✅")
            st.rerun()  # Refresh sidebar counter

    else:
        st.info(
            "📂 Upload a photo to begin. "
            "The AI will classify the debris type and let you log a geo-tagged report."
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Pollution Map
# ══════════════════════════════════════════════════════════════════════════════
with tab_map:
    st.subheader("🗺️ Geospatial Pollution Hotspots")

    df = get_reports_df()

    if df.empty:
        st.info(
            "📭 No reports in the database yet. "
            "Submit a report in the **📸 Scan & Report** tab and it will appear here!"
        )
        # Still render a default map so the page doesn't look broken
        m = folium.Map(
            location=[20.5937, 78.9629],  # Centre of India
            zoom_start=5,
            tiles="CartoDB dark_matter",
        )
        st_folium(m, use_container_width=True, height=500)

    else:
        # Map centred on the average of all report locations
        avg_lat = df["lat"].mean()
        avg_lon = df["lon"].mean()

        m = folium.Map(
            location=[avg_lat, avg_lon],
            zoom_start=6,
            tiles="CartoDB dark_matter",
        )

        for _, row in df.iterrows():
            colour = MARKER_COLOURS.get(row["severity"], "blue")
            popup_html = f"""
            <b>Report #{int(row['id'])}</b><br>
            Type: <b>{row['predicted_class'].capitalize()}</b><br>
            Confidence: {float(row['confidence']):.0%}<br>
            Severity: <b>{row['severity']}</b><br>
            <small>{row['timestamp']}</small>
            """
            folium.Marker(
                location=[row["lat"], row["lon"]],
                popup=folium.Popup(popup_html, max_width=220),
                tooltip=f"{row['predicted_class'].capitalize()} — {row['severity']}",
                icon=folium.Icon(color=colour, icon="exclamation-sign"),
            ).add_to(m)

        map_col, stats_col = st.columns([2, 1], gap="large")

        with map_col:
            st_folium(m, use_container_width=True, height=520)

        with stats_col:
            st.markdown("#### 📊 Quick Stats")
            st.metric("Total Reports", len(df))

            st.markdown("**By Severity**")
            sev_counts = df["severity"].value_counts()
            for sev, cnt in sev_counts.items():
                colour = SEVERITY_COLOURS.get(sev, "#888")
                st.markdown(
                    f"<span style='color:{colour}; font-weight:700;'>● {sev}</span>  "
                    f"<span style='color:#c9d1d9;'>{cnt} reports</span>",
                    unsafe_allow_html=True,
                )

            st.markdown("**By Debris Type**")
            class_counts = df["predicted_class"].value_counts()
            for cls, cnt in class_counts.items():
                st.markdown(f"- `{cls}`: **{cnt}**")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Performance & About
# ══════════════════════════════════════════════════════════════════════════════
with tab_metrics:
    st.subheader("📊 Technical Architecture & Evaluation")

    st.markdown(
        """
        ### System Architecture
        ```
        User Photo
            ↓
        EfficientNetB0 (Transfer Learning, Fine-tuned)
            ↓
        [Predicted Class + Confidence Score]
            ↓
        Grad-CAM Heatmap  ──→  Visual Explanation Overlay
            ↓
        Severity Lookup Table  ──→  Priority Label
            ↓
        LLM API  ──→  Plain-language Ecological Impact Explanation
            ↓
        SQLite + Folium  ──→  Logged & Mapped as Geospatial Hotspot
        ```

        ### Core Technologies
        | Component | Technology |
        |---|---|
        | **Deep Learning Model** | EfficientNetB0 (Transfer Learning via TensorFlow/Keras) |
        | **Explainability** | Grad-CAM Class Activation Maps |
        | **Frontend** | Streamlit (Python-native, multi-tab) |
        | **Geospatial Mapping** | Folium + streamlit-folium |
        | **Persistent Storage** | SQLite (local, zero-config, portable) |
        | **LLM Narration** | Gemini API (Day 10) |
        | **Deployment** | Hugging Face Spaces (Day 14) |

        ### Target Debris Classes
        | Class | Severity | Ecological Rationale |
        |---|---|---|
        | Plastic | 🔴 **Critical** | Non-biodegradable; microplastic generation harms marine food chains |
        | Metal | 🟠 **High** | Heavy metal leaching and physical laceration hazard |
        | Glass | 🟠 **High** | Habitat destruction and laceration risk for aquatic life |
        | Cardboard | 🔵 **Medium** | Blocks drainage; decomposes slowly in water |
        | Paper | 🟢 **Low** | Rapid organic degradation, lower long-term risk |
        | Trash (Mixed) | 🔵 **Medium** | Heterogeneous litter requiring varied remediation |

        ---
        *Model performance metrics (confusion matrix, per-class F1, training curves)
        will be displayed here once Person A's EfficientNetB0 training is complete on Day 7.*
        """
    )
