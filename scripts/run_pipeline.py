"""
run_pipeline.py — NASDAQ Quant Analyzer 진입점
"""
import logging
import os
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from orchestrator.pipeline import Pipeline

logging.basicConfig(
    level=logging.DEBUG if os.getenv("DEBUG", "false").lower() == "true" else logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)

if __name__ == "__main__":
    model = os.getenv("MODEL", "claude-sonnet-4-20250514")
    top_n = int(os.getenv("TOP_N", "10"))
    pipeline = Pipeline(model=model, top_n=top_n)
    pipeline.run()
