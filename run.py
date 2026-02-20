import subprocess
import sys
import os

from dotenv import load_dotenv
load_dotenv()

def main() -> None:
    # Equivalent to: streamlit run app.py
    cmd = [sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.address=0.0.0.0",
        "--server.port=8501",
        "--server.headless=true",
    ]
    raise SystemExit(subprocess.call(cmd))

if __name__ == "__main__":
    main()