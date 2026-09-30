"""
AquaScan — AI Waterway Pollution Reporter
Root entrypoint for Hugging Face Spaces and cloud deployment.
Delegates directly to app/app.py.
"""

import sys
import runpy
from pathlib import Path

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Run main application
runpy.run_path(str(ROOT / "app" / "app.py"), run_name="__main__")
