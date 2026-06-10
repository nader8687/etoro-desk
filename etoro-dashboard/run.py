"""Container entrypoint — health server + Streamlit."""
from __future__ import annotations

import health_server

health_server.start_background()

from streamlit.web import cli as stcli
import sys

sys.argv = [
    "streamlit",
    "run",
    "app.py",
    "--server.port=8501",
    "--server.address=0.0.0.0",
    "--server.headless=true",
    "--browser.gatherUsageStats=false",
]
stcli.main()
