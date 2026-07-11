"""Minimal backend usage of the stable ml_service façade."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_service import AccidentRiskPredictor

predictor = AccidentRiskPredictor()
city_result = predictor.predict_city(datetime_hour="2022-09-08 15:00:00", horizon="1h")
segment_result = predictor.predict_segment(
    road_segment_id="2744171408_2744219355_0",
    datetime_hour="2022-09-08 15:00:00",
    horizon="24h",
)
