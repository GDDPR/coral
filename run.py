import subprocess
import sys

def main() -> None:
    # Equivalent to: streamlit run app.py
    cmd = [sys.executable, "-m", "streamlit", "run", "app.py"]
    raise SystemExit(subprocess.call(cmd))

if __name__ == "__main__":
    main()