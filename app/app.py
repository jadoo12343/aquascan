"""
AquaScan — AI Waterway Pollution Reporter
Frontend & Geospatial Mapping Interface (Person B)
"""

import streamlit as st

st.set_page_config(
    page_title="AquaScan — Waterway Pollution Reporter",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header Section
st.title("🌊 AquaScan")
st.caption("AI-Powered Aquatic Debris Detection, Explainability & Geospatial Hotspot Mapping")

# Sidebar
with st.sidebar:
    st.header("⚙️ System Status")
    st.success("Frontend: Streamlit Active")
    st.info("Storage: SQLite (Local)")
    st.write("---")
    st.markdown("### 📌 Quick Guide")
    st.markdown(
        """
        1. **Scan & Report:** Upload a photo of waterway debris.
        2. **Inspect AI Decision:** View predicted class + Grad-CAM heatmap.
        3. **Explore Hotspots:** Review logged reports on the interactive map.
        """
    )

# Tab Navigation
tab_scan, tab_map, tab_metrics = st.tabs([
    "📸 Scan & Report",
    "🗺️ Pollution Map",
    "📊 Model Performance & About"
])

with tab_scan:
    st.subheader("Report Aquatic Debris")
    uploaded_file = st.file_uploader(
        "Choose an image of waste found near a waterway...",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        with col2:
            st.info("Inference pipeline wiring ready for Day 3-4 model checkpoint.")
            st.metric(label="Status", value="Image Loaded Successfully")
    else:
        st.info("Upload a photo or choose a sample image to begin classification.")

with tab_map:
    st.subheader("Geospatial Pollution Hotspots")
    st.info("Interactive Folium map initialized. Data logging activates on Day 10-12.")

with tab_metrics:
    st.subheader("Technical Architecture & Evaluation")
    st.markdown(
        """
        - **Core Model:** Fine-Tuned EfficientNetB0 (Transfer Learning)
        - **Explainability:** Grad-CAM Class Activation Maps
        - **Target Categories:** Plastic, Metal, Glass, Cardboard, Paper, General Trash
        - **Impact:** Geospatial clustering & severity prioritization for cleanup teams
        """
    )
