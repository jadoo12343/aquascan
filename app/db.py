"""
AquaScan — SQLite Database Layer (Person B)
Handles all persistent storage for pollution reports.

Schema:
    reports(id, image_path, predicted_class, confidence, severity, lat, lon, timestamp)
"""

import sqlite3
import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path

# ── Path to the database file (lives in data/ which is git-ignored for large files)
DB_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DB_DIR / "aquascan.db"


def _get_connection() -> sqlite3.Connection:
    """Open a thread-safe SQLite connection with row factory enabled."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False, timeout=10.0)
    conn.row_factory = sqlite3.Row   # Allows dict-style access: row["lat"]
    return conn


def init_db() -> None:
    """
    Create the reports table if it doesn't already exist.
    Safe to call on every app startup — uses CREATE TABLE IF NOT EXISTS.
    """
    conn = _get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS reports (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path      TEXT    NOT NULL,
            predicted_class TEXT    NOT NULL,
            confidence      REAL    NOT NULL,
            severity        TEXT    NOT NULL,
            lat             REAL    NOT NULL,
            lon             REAL    NOT NULL,
            timestamp       DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


def add_report(
    image_path: str,
    predicted_class: str,
    confidence: float,
    severity: str,
    lat: float,
    lon: float,
) -> int:
    """
    Insert a new pollution report into the database.

    Args:
        image_path      : Local path where the uploaded image was saved.
        predicted_class : Model output label, e.g. 'plastic'.
        confidence      : Prediction probability 0.0–1.0.
        severity        : Human-readable risk tier ('Critical','High','Medium','Low').
        lat             : GPS latitude (decimal degrees).
        lon             : GPS longitude (decimal degrees).

    Returns:
        The newly inserted row's id (integer).
    """
    conn = _get_connection()
    cursor = conn.execute(
        """
        INSERT INTO reports (image_path, predicted_class, confidence, severity, lat, lon)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (image_path, predicted_class, round(confidence, 4), severity, lat, lon),
    )
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return new_id


def get_all_reports() -> list[dict]:
    """
    Fetch all reports as a list of plain Python dicts, newest first.
    Returns an empty list if no reports exist yet.
    """
    conn = _get_connection()
    rows = conn.execute(
        "SELECT * FROM reports ORDER BY timestamp DESC"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_reports_df() -> pd.DataFrame:
    """
    Return all reports as a Pandas DataFrame — ready for Folium map
    rendering and Seaborn / Matplotlib charts in the metrics tab.

    Returns:
        pd.DataFrame with columns:
            id, image_path, predicted_class, confidence,
            severity, lat, lon, timestamp
    """
    reports = get_all_reports()
    if not reports:
        # Return an empty DataFrame with the correct column names
        return pd.DataFrame(
            columns=[
                "id", "image_path", "predicted_class",
                "confidence", "severity", "lat", "lon", "timestamp",
            ]
        )
    return pd.DataFrame(reports)


def get_report_count() -> int:
    """Return the total number of reports in the database."""
    conn = _get_connection()
    count = conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0]
    conn.close()
    return count


def delete_report(report_id: int) -> bool:
    """Delete a single report by its id. Returns True if a row was deleted."""
    conn = _get_connection()
    cursor = conn.execute("DELETE FROM reports WHERE id = ?", (report_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


def clear_all_reports() -> None:
    """Remove all records from the reports table."""
    conn = _get_connection()
    conn.execute("DELETE FROM reports")
    conn.commit()
    conn.close()


def get_severity_summary() -> dict[str, int]:
    """
    Return counts grouped by severity in priority order.
    Guarantees all four keys exist even if count is 0.
    """
    counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
    conn = _get_connection()
    rows = conn.execute(
        "SELECT severity, COUNT(*) as count FROM reports GROUP BY severity"
    ).fetchall()
    conn.close()
    for row in rows:
        sev = row["severity"]
        if sev in counts:
            counts[sev] = row["count"]
    return counts


def get_stats_summary() -> dict:
    """
    Return high-level summary metrics across all reports for dashboard KPIs:
    - total_reports: int
    - high_risk_pct: float (Critical + High share, 0.0–1.0)
    - top_class: str
    - last_report_time: str or None
    """
    df = get_reports_df()
    if df.empty:
        return {
            "total_reports": 0,
            "high_risk_pct": 0.0,
            "top_class": "N/A",
            "last_report_time": None,
        }
    
    total = len(df)
    high_risk_count = len(df[df["severity"].isin(["Critical", "High"])])
    high_risk_pct = round(high_risk_count / total, 3) if total > 0 else 0.0
    top_class = df["predicted_class"].mode().iloc[0] if not df["predicted_class"].empty else "N/A"
    last_time = df["timestamp"].iloc[0] if "timestamp" in df.columns and not df.empty else None

    return {
        "total_reports": total,
        "high_risk_pct": high_risk_pct,
        "top_class": str(top_class).capitalize(),
        "last_report_time": last_time,
    }


def export_geojson(df: pd.DataFrame = None) -> str:
    """
    Export reports to an RFC 7946 compliant GeoJSON FeatureCollection string,
    enabling direct drag-and-drop ingestion into municipal GIS tools (QGIS, ArcGIS, Mapbox).
    """
    if df is None:
        df = get_reports_df()

    features = []
    for _, row in df.iterrows():
        try:
            lat = float(row["lat"])
            lon = float(row["lon"])
        except (ValueError, TypeError):
            continue

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [round(lon, 6), round(lat, 6)],  # GeoJSON standard is [longitude, latitude]
            },
            "properties": {
                "id": int(row["id"]),
                "predicted_class": str(row["predicted_class"]),
                "confidence": round(float(row["confidence"]), 4),
                "severity": str(row["severity"]),
                "timestamp": str(row["timestamp"]) if pd.notna(row.get("timestamp")) else "",
            },
        }
        features.append(feature)

    collection = {
        "type": "FeatureCollection",
        "name": "AquaScan_Waterway_Pollution_Reports",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
        },
        "features": features,
    }
    return json.dumps(collection, indent=2)


def seed_demo_reports(force: bool = False) -> int:
    """
    Seed realistic waterway debris reports for live demonstration and judging.
    Only seeds if the database is currently empty, unless force=True.
    Returns the number of seeded records.
    """
    if not force and get_report_count() > 0:
        return 0

    demo_data = [
        # (image_stub, class, confidence, severity, lat, lon)
        ("data/uploads/seed_plastic_yamuna.jpg", "plastic", 0.94, "Critical", 28.6139, 77.2090),
        ("data/uploads/seed_plastic_yamuna2.jpg", "plastic", 0.91, "Critical", 28.6185, 77.2140),
        ("data/uploads/seed_metal_yamuna.jpg", "metal", 0.88, "High", 28.6110, 77.2065),
        ("data/uploads/seed_plastic_mithi.jpg", "plastic", 0.96, "Critical", 19.0760, 72.8777),
        ("data/uploads/seed_trash_mithi.jpg", "trash", 0.83, "Medium", 19.0795, 72.8720),
        ("data/uploads/seed_glass_juhu.jpg", "glass", 0.89, "High", 19.0990, 72.8265),
        ("data/uploads/seed_plastic_juhu.jpg", "plastic", 0.92, "Critical", 19.1030, 72.8240),
        ("data/uploads/seed_cardboard_varanasi.jpg", "cardboard", 0.85, "Medium", 25.3176, 83.0062),
        ("data/uploads/seed_plastic_varanasi.jpg", "plastic", 0.95, "Critical", 25.3140, 83.0110),
        ("data/uploads/seed_paper_sabarmati.jpg", "paper", 0.81, "Low", 23.0225, 72.5714),
        ("data/uploads/seed_plastic_sabarmati.jpg", "plastic", 0.90, "Critical", 23.0280, 72.5760),
        ("data/uploads/seed_metal_chilika.jpg", "metal", 0.87, "High", 19.7147, 85.3220),
        ("data/uploads/seed_trash_kochi.jpg", "trash", 0.84, "Medium", 9.9312, 76.2673),
        ("data/uploads/seed_plastic_kochi.jpg", "plastic", 0.93, "Critical", 9.9380, 76.2610),
    ]

    conn = _get_connection()
    for item in demo_data:
        conn.execute(
            """
            INSERT INTO reports (image_path, predicted_class, confidence, severity, lat, lon)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            item,
        )
    conn.commit()
    conn.close()
    return len(demo_data)
