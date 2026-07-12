"""Collect bounded TomTom flow snapshots for the highest model-risk segments."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_service import AccidentRiskPredictor, TomTomTrafficService


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datetime", required=True, help="Model-supported hour, e.g. 2022-09-08T15:00:00")
    parser.add_argument("--horizon", choices=("1h", "24h"), default="1h")
    parser.add_argument("--top-n", type=int, default=int(os.getenv("TOMTOM_TRAFFIC_TOP_N", "100")))
    parser.add_argument("--interval-seconds", type=int, default=int(os.getenv("TOMTOM_TRAFFIC_INTERVAL_SECONDS", "3600")))
    parser.add_argument("--iterations", type=int, default=1, help="Number of bounded collection runs; 1 is safest for manual use.")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "live_traffic" / "tomtom_flow_snapshots.parquet")
    args = parser.parse_args()
    if not 1 <= args.top_n <= 100:
        parser.error("--top-n must be between 1 and 100 to keep traffic collection bounded")
    if args.interval_seconds < 1 or args.iterations < 1:
        parser.error("--interval-seconds and --iterations must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    risk = AccidentRiskPredictor().predict_city(args.datetime, args.horizon)["predictions"]
    selected = sorted(risk, key=lambda item: item["risk_probability"], reverse=True)[:args.top_n]
    traffic = TomTomTrafficService()
    for iteration in range(args.iterations):
        readings = traffic.collect((item["road_segment_id"] for item in selected), args.output)
        available = sum(item["available"] for item in readings)
        logging.info("Collected %d/%d live traffic readings into %s", available, len(readings), args.output)
        if iteration + 1 < args.iterations:
            time.sleep(args.interval_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
