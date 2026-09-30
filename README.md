---
title: AquaScan — Waterway Pollution Reporter
emoji: 🌊
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.38.0
app_file: app.py
pinned: false
license: mit
---

# AquaScan — AI Waterway Pollution Reporter

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![HuggingFace Spaces](https://img.shields.io/badge/Deploy-HuggingFace%20Spaces-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

> **Devpost ML Empowerment Build Challenge** | Category: **Sustainability AI**  
> An explainable, geospatial AI platform enabling community volunteers, conservation NGOs, and municipal teams to detect, classify, and prioritize aquatic pollution before it enters open oceans.

---

## 1. Problem Statement
Over 8 million metric tons of waste enter aquatic ecosystems annually, yet municipal cleanup teams and local conservation groups lack accessible, granular data to detect and prioritize pollution hotspots before debris reaches open water. Existing reporting workflows rely on generic citizen complaints without standardized classification, objective severity assessment, or actionable visual evidence, resulting in delayed responses and misallocated cleanup resources.

## 2. Target Users
- **Community Environmental Volunteers:** Citizens photographing debris along riverbanks, lakeshores, and beaches who need an instant, smartphone-accessible tool that classifies waste and logs geo-tagged reports in under 30 seconds.
- **Watershed Conservation NGOs:** Organizations tracking pollution patterns across seasons and prioritizing cleanup drives using data-backed hotspot analytics.
- **Municipal Waste Management & Cleanup Coordinators:** Municipal teams receiving centralized dashboards with explainable AI validation (Grad-CAM heatmaps) to verify reports and dispatch crews to high-risk areas.

---

## 3. Core Architecture
```
[User Image]
     │
     ▼
[Transfer Learning Classifier] (EfficientNetB0) ───► [Predicted Waste Class & Confidence]
     │                                                           │
     ├──► [Grad-CAM Explainability] ──► [Heatmap Overlay]        ├──► [Severity Scoring]
     │                                                           │          │
     │                                                           ▼          ▼
     └───────────────────────────────────────────────────► [LLM Narration Layer] (Groq/Gemini)
                                                                 │
                                                                 ▼
                                                    [SQLite DB + Folium Geo-Map]
```

---

## 4. Repository Structure
```
aquascan/
├── .gitignore            # Ignores datasets, venv, model weights, API secrets
├── README.md             # Project documentation, setup guide & architecture
├── requirements.txt      # Application & ML dependencies (TensorFlow, Streamlit, Folium)
├── app.py                # Hugging Face Spaces entrypoint shim
├── app/                  # Frontend, geospatial mapping & database logic
│   ├── app.py            # Streamlit dashboard (Scan, Map, Model Metrics, Export)
│   └── db.py             # SQLite persistence, spatial aggregation & GeoJSON export
├── model/                # Model inference, Grad-CAM, training & evaluation
│   ├── predict.py        # EfficientNetB0 inference + Top-3 probability calibration
│   ├── gradcam.py        # Grad-CAM heatmap extraction from 'top_conv' layer
│   ├── train.py          # 2-Phase transfer learning pipeline (frozen + fine-tune)
│   ├── prepare_dataset.py# Merges & formats Kaggle/TrashNet/TACO datasets
│   ├── severity.py       # Ecological risk scoring matrix per debris class
│   ├── narration.py      # Groq LLaMA 3.1 ecological guidance layer
│   ├── eval.py           # Benchmark metrics, confusion matrix & Grad-CAM lab
│   ├── mock.py           # Offline fallback mock generator
│   └── weights/          # Storage for efficientnetb0_aquascan.h5 (git-ignored)
├── notebooks/            # Jupyter training pipelines & EDA
│   └── training_pipeline.ipynb # Google Colab T4-ready training pipeline
└── data/                 # SQLite database & uploaded images (git-ignored)
    ├── aquascan.db       # Seeded pollution reports database
    └── uploads/          # Geo-tagged debris images
```

---

## 5. Local Setup Instructions

### 1. Clone & Navigate
```bash
git clone <repo-url>
cd aquascan
```

### 2. Set Up Virtual Environment
On Windows:
```powershell
python -m venv venv
.\venv\Scripts\activate
```

On macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment (Optional for AI Narration)
To enable real-time LLaMA 3.1 ecological impact explanations, set a free Groq API key:
```bash
# Windows (PowerShell)
$env:GROQ_API_KEY="your-groq-api-key"

# Linux / macOS
export GROQ_API_KEY="your-groq-api-key"
```
*(Note: If omitted, AquaScan automatically falls back to standard rule-based ecological guidance without errors).*

### 5. Run the Streamlit Application
```bash
streamlit run app.py
# or
streamlit run app/app.py
```

### 6. Cloud Deployment (Hugging Face Spaces)
AquaScan is pre-configured for instant zero-cost hosting on Hugging Face Spaces:
1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space) and select **Streamlit** SDK.
2. Link your GitHub repository (`main` branch) or push directly to the Space's Git remote.
3. *(Optional)* In **Settings > Variables and secrets**, add `GROQ_API_KEY` as a secret for real-time LLaMA 3.1 ecological guidance.
4. The Space will automatically detect the YAML frontmatter, install `requirements.txt`, and launch `app.py`.

---

## 6. 🚀 60-Second Judge Demo Tour

If you are evaluating AquaScan, you can test the full end-to-end pipeline in under a minute without uploading your own images:

1. **Scan Tab (Classification & Inference):**
   - Click one of the instant demo buttons (e.g., `🧴 Plastic Bottle` or `🥫 Metal Can`).
   - Observe the **EfficientNetB0 classification card** and the **Top-3 Confidence probability breakdown**.
   - Notice the **Groq LLaMA 3.1 Ecological Guidance** explaining environmental degradation and toxicity.
   - Click the map to drop a pin or pick a waterway preset (e.g., *Yamuna River*), then click **✅ Save Report to Database**.

2. **Pollution Map Tab (Geospatial Analytics):**
   - Switch to the **🗺️ Pollution Map** tab.
   - View the interactive clustered markers and dynamic heatmap showing real-time pollution density.
   - Click any pin to inspect the original photo, detected debris category, confidence score, and timestamp.

3. **Model & Metrics Tab (Explainability & Responsible AI):**
   - Switch to **📊 Model & Metrics**.
   - Explore the **1,200-sample test set confusion matrix** and per-class Precision/Recall/F1 scores.
   - Switch to the **Explainability (Grad-CAM)** sub-tab to view real-time class activation heatmaps showing which pixels the convolutional layers focused on.
   - Download the official **Model Card (`aquascan_model_card.json`)**.

---

## 7. Architecture & Frontend Decision
- **Frontend Framework:** Streamlit was chosen over Gradio for AquaScan because interactive geospatial mapping (`streamlit-folium`) is a core feature for the demo and evaluation criteria (Real-World Impact 20%, UX 15%). Streamlit allows seamless integration of multi-tab layouts (Report, Map, Model Metrics) with stateful SQLite data handling in pure Python.

---

## 7. Dataset & Storage Convention

> **Important for anyone cloning this repo:** The `data/` directory is git-ignored to prevent large image files from bloating the repository. Follow the instructions below to source the dataset locally.

### Training Data Sources (Person A)
The AquaScan classifier is trained on a merged dataset assembled from three public sources:

| Dataset | Images | Source |
|---|---|---|
| Kaggle Garbage Classification | ~2,500 | [Kaggle](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification) |
| TrashNet | ~2,500 | [GitHub](https://github.com/garythung/trashnet) |
| TACO (Trash Annotations in Context) | ~1,500 (cropped) | [TACO Dataset](http://tacodataset.org/) |

**Target classes:** `plastic`, `metal`, `glass`, `cardboard`, `paper`, `trash`

### Local Directory Layout (after setup)
```
aquascan/
├── data/
│   ├── aquascan.db          ← SQLite database (auto-created on first run)
│   ├── uploads/             ← User-submitted photos (saved by app at runtime)
│   ├── raw/                 ← Downloaded original dataset images (set up manually)
│   └── processed/           ← Pre-processed & augmented images (Person A pipeline)
├── model/
│   ├── mock.py              ← Mock classifier stub (Days 1–6, Person B)
│   └── predict.py           ← EfficientNetB0 inference (Day 7+, Person A delivers)
└── app/
    ├── app.py               ← Streamlit application
    └── db.py                ← SQLite database engine
```

### Setting Up Data Locally
1. Download datasets from the sources above.
2. Place raw images into `data/raw/` organized by class folder.
3. Run Person A's preprocessing notebook in `notebooks/` to generate `data/processed/`.
4. The `data/uploads/` folder and `data/aquascan.db` are created automatically when you run the app.

---

## 8. Model Evaluation & Explainability (Grad-CAM)

AquaScan includes an in-app empirical evaluation suite (Tab 3) adhering to responsible AI reporting:
- **Test Set Size:** 1,200 curated test images across 6 target classes.
- **Confusion Matrix:** Interactive heatmap displaying true vs. predicted classifications with normalized (%) and count views.
- **Explainability:** Grad-CAM Class Activation Mapping targeting the final convolutional layer (`top_conv`) of EfficientNetB0, verifying that feature activations isolate debris contours rather than background river currents or terrain.
- **Governance:** In-app downloadable Model Evaluation Card (`aquascan_model_card.json`).
