"""
AquaScan — SQLite Database Layer (Person B)
Handles all persistent storage for pollution reports.

Schema:
    reports(id, image_path, predicted_class, confidence, severity, lat, lon, timestamp)
"""

import sqlite3
import os
import pandas as pd
from datetime import datetime
from pathlib import Path

# ── Path to the database file (lives in data/ which is git-ignored for large files)
DB_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DB_DIR / "aquascan.db"


def _get_connection() -> sqlite3.Connection:
    """Open a thread-safe SQLite connection with row factory enabled."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
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


def delete_report(report_id: int) -> None:
    """Delete a single report by its id. Used for admin cleanup."""
    conn = _get_connection()
    conn.execute("DELETE FROM reports WHERE id = ?", (report_id,))
    conn.commit()
    conn.close()
