import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs/formal")
    p.add_argument("--data", default="data")
    args = p.parse_args()
    frozen = json.loads((Path(args.runs) / "frozen.json").read_text(encoding="utf-8"))
    checkpoint = Path(args.runs) / frozen["selection"]["run_id"] / "best.pt"
    app = Path(__file__).resolve().parents[1] / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app), "--server.address", "127.0.0.1", "--",
                    "--checkpoint", str(checkpoint), "--data", args.data], check=True)


if __name__ == "__main__":
    main()
