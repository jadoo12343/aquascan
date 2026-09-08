# AquaScan — AI Waterway Pollution Reporter

> **ML Empowerment Build Challenge** | Target Category: **Sustainability AI**  
> An explainable, geospatial AI platform enabling community volunteers and municipal teams to detect, classify, and prioritize aquatic pollution before it enters open waterways.

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
├── .gitignore            # Ignores large datasets, venv, model weights, secrets
├── README.md             # Project documentation & setup instructions
├── requirements.txt      # Python dependencies for the application
├── app/                  # Frontend, mapping & backend logic (Person B)
│   └── app.py            # Streamlit dashboard & Folium map integration
├── model/                # Model training, evaluation & Grad-CAM scripts (Person A)
├── notebooks/            # Exploratory data analysis & prototyping notebooks
└── data/                 # Raw and processed datasets (git-ignored)
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

### 4. Run the Streamlit Application
```bash
streamlit run app/app.py
```

---

## 6. Architecture & Frontend Decision
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
