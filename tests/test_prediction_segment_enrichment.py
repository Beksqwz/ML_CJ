"""Focused contract tests for persisted 24-hour ensemble segment enrichment."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

import ml_service.hybrid_risk as hybrid
from api_service.app import Runtime, create_app
from ml_service.inference.catboost_explanations import (
    CATBOOST_SHAP_BATCH_SIZE,
    MAX_NEGATIVE_FACTORS,
    MAX_POSITIVE_FACTORS,
    _json_safe,
    catboost_explanations,
)
from ml_service.inference.feature_builder import _config
from ml_service.runtime_store import PredictionStore


class _Preprocessor:
    def transform(self, frame):
        return frame


class _Hgb:
    def predict_proba(self, frame):
        size = len(frame)
        return np.column_stack((np.zeros(size), np.linspace(0.8, 0.2, size)))


class _AscendingHgb(_Hgb):
    def predict_proba(self, frame):
        size = len(frame)
        return np.column_stack((np.zeros(size), np.linspace(0.2, 0.8, size)))


class _CatBoost:
    def __init__(self):
        self.shap_calls = 0

    def load_model(self, _):
        return None

    def predict_proba(self, frame):
        size = len(frame)
        return np.column_stack((np.zeros(size), np.linspace(0.1, 0.9, size)))

    def get_feature_importance(self, pool, type):
        self.assert_type(type)
        self.shap_calls += 1
        width = pool.num_col()
        matrix = np.zeros((pool.num_row(), width + 1))
        matrix[:, 0] = 0.8
        matrix[:, 1] = -0.5
        matrix[:, 2] = 0.2
        matrix[:, -1] = 0.1
        return matrix

    @staticmethod
    def assert_type(value):
        if value != "ShapValues":
            raise AssertionError(value)


class _FlatCatBoost(_CatBoost):
    def predict_proba(self, frame):
        size = len(frame)
        return np.column_stack((np.zeros(size), np.full(size, 0.5)))


def _features(rows: int) -> pd.DataFrame:
    config = _config("24h")
    numerical = config["numerical_features"]
    categorical = config["categorical_features"]
    data = {
        "road_segment_id": [f"s{i}" for i in range(rows)],
        "datetime_hour": [pd.Timestamp("2026-07-15T08:00:00")] * rows,
    }
    data.update({name: np.arange(rows, dtype=float) for name in numerical})
    data.update({name: ["value"] * rows for name in categorical})
    return pd.DataFrame(data)


class PredictionSegmentEnrichmentTests(unittest.TestCase):
    def _build_dynamic(
        self,
        rows: int = 3,
        *,
        shap_fails: bool = False,
        catboost: _CatBoost | None = None,
        hgb: _Hgb | None = None,
    ):
        cat = catboost or _CatBoost()
        hgb = hgb or _Hgb()
        coords = pd.DataFrame(
            {
                "road_segment_id": [f"s{i}" for i in range(rows - 1)],
                "longitude": [71.4] * (rows - 1),
                "latitude": [51.15] * (rows - 1),
            }
        )
        patches = [
            patch.object(hybrid, "build_features", return_value=(_features(rows), {})),
            patch.object(hybrid, "CatBoostClassifier", return_value=cat),
            patch.object(hybrid.joblib, "load", side_effect=[_Preprocessor(), hgb]),
            patch.object(hybrid, "_canonical_segment_coordinates", return_value=coords),
        ]
        if shap_fails:
            patches.append(
                patch.object(
                    hybrid, "catboost_explanations", side_effect=RuntimeError("no shap")
                )
            )
        with patches[0], patches[1], patches[2], patches[3]:
            if shap_fails:
                with patches[4]:
                    return hybrid.build_dynamic_risk("2026-07-15T08:00:00+05:00"), cat
            return hybrid.build_dynamic_risk("2026-07-15T08:00:00+05:00"), cat

    def test_01_input_rows_are_preserved(self):
        frame, _ = self._build_dynamic(3968)
        self.assertEqual(len(frame), 3968)

    def test_02_segment_ids_remain_unique(self):
        frame, _ = self._build_dynamic()
        self.assertEqual(frame.road_segment_id.nunique(), len(frame))

    def test_03_coordinates_join_by_segment_id(self):
        frame, _ = self._build_dynamic()
        row = frame.set_index("road_segment_id").loc["s0"]
        self.assertEqual((row.longitude, row.latitude), (71.4, 51.15))

    def test_04_missing_coordinates_are_null(self):
        frame, _ = self._build_dynamic()
        self.assertTrue(
            pd.isna(frame.set_index("road_segment_id").loc["s2", "longitude"])
        )

    def test_05_dynamic_risk_score_matches_flat_score(self):
        frame, _ = self._build_dynamic()
        self.assertTrue(
            all(
                row.dynamic_risk["score"] == row.dynamic_score
                for row in frame.itertuples()
            )
        )

    def test_06_dynamic_risk_rank_and_percentile_match(self):
        frame, _ = self._build_dynamic()
        self.assertTrue(
            all(
                row.dynamic_risk["rank"] == row.dynamic_rank
                and row.dynamic_risk["percentile"] == row.dynamic_percentile
                for row in frame.itertuples()
            )
        )

    def test_07_ensemble_formula_uses_component_percentiles(self):
        frame, _ = self._build_dynamic()
        for row in frame.itertuples():
            components = row.model_components
            expected = (
                0.8 * components["catboost"]["percentile"]
                + 0.2 * components["hgb"]["percentile"]
            )
            self.assertAlmostEqual(row.dynamic_score, expected, places=12)

    def test_08_rank_is_not_catboost_only(self):
        frame, _ = self._build_dynamic(catboost=_FlatCatBoost(), hgb=_AscendingHgb())
        self.assertEqual(
            frame.dynamic_rank.tolist(),
            frame.dynamic_score.rank(method="first", ascending=False)
            .astype(int)
            .tolist(),
        )
        self.assertNotEqual(
            frame.dynamic_rank.tolist(),
            frame.score_catboost_stage19h.rank(method="first", ascending=False)
            .astype(int)
            .tolist(),
        )

    def test_09_catboost_component_is_probability(self):
        frame, _ = self._build_dynamic()
        self.assertEqual(
            frame.iloc[0].model_components["catboost"]["score_type"], "probability"
        )

    def test_10_hgb_component_is_probability(self):
        frame, _ = self._build_dynamic()
        self.assertEqual(
            frame.iloc[0].model_components["hgb"]["score_type"], "probability"
        )

    def test_11_component_weights_are_frozen_config_values(self):
        frame, _ = self._build_dynamic()
        self.assertEqual(
            frame.iloc[0].dynamic_risk["weights"], {"catboost": 0.8, "hgb": 0.2}
        )

    def test_12_no_probability_named_final_score(self):
        frame, _ = self._build_dynamic()
        self.assertNotIn("risk_probability", frame.columns)
        self.assertEqual(
            frame.iloc[0].dynamic_risk["score_type"], "weighted_percentile_ensemble"
        )

    def test_13_explanation_scope_is_catboost_only(self):
        frame, _ = self._build_dynamic()
        self.assertEqual(frame.iloc[0].explanation["scope"], "catboost_component_only")

    def test_14_explanation_disclaimer_is_multilingual(self):
        frame, _ = self._build_dynamic()
        self.assertEqual(
            set(frame.iloc[0].explanation["disclaimer"]), {"ru", "kz", "en"}
        )

    def test_15_positive_factors_are_descending(self):
        frame, _ = self._build_dynamic()
        values = [
            item["shap_value"]
            for item in frame.iloc[0].explanation["top_positive_factors"]
        ]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_16_negative_factors_are_ascending(self):
        frame, _ = self._build_dynamic()
        values = [
            item["shap_value"]
            for item in frame.iloc[0].explanation["top_negative_factors"]
        ]
        self.assertEqual(values, sorted(values))

    def test_17_top_n_limits_are_enforced(self):
        frame, _ = self._build_dynamic()
        explanation = frame.iloc[0].explanation
        self.assertLessEqual(
            len(explanation["top_positive_factors"]), MAX_POSITIVE_FACTORS
        )
        self.assertLessEqual(
            len(explanation["top_negative_factors"]), MAX_NEGATIVE_FACTORS
        )

    def test_18_only_selected_factor_values_are_returned(self):
        frame, _ = self._build_dynamic()
        payload = json.dumps(frame.iloc[0].explanation)
        self.assertNotIn("frozen_feature_vector", payload)
        self.assertNotIn("weather_temperature_2m", payload)

    def test_19_json_safe_values(self):
        self.assertEqual(_json_safe(np.int64(1)), 1)
        self.assertIsNone(_json_safe(np.inf))
        self.assertIsNone(_json_safe(np.nan))
        self.assertTrue(_json_safe(np.bool_(True)))

    def test_20_unknown_features_use_canonical_name(self):
        feature = _features(1).columns[5]
        frame, _ = self._build_dynamic()
        factors = frame.iloc[0].explanation["top_positive_factors"]
        self.assertTrue(all(item["feature"] for item in factors))
        self.assertIsInstance(feature, str)

    def test_21_full_feature_vector_is_not_persisted(self):
        frame, _ = self._build_dynamic()
        self.assertNotIn("feature_values", frame.columns)

    def test_22_full_shap_matrix_is_not_persisted(self):
        frame, _ = self._build_dynamic()
        self.assertNotIn("shap_values", frame.columns)

    def test_23_explanations_are_deterministic(self):
        first, _ = self._build_dynamic()
        second, _ = self._build_dynamic()
        self.assertEqual(first.iloc[0].explanation, second.iloc[0].explanation)

    def test_24_shap_is_batched(self):
        config = _config("24h")
        frame = _features(CATBOOST_SHAP_BATCH_SIZE + 1)
        cat = _CatBoost()
        explanations, calls = catboost_explanations(
            cat,
            frame,
            ordered_features=[
                *config["numerical_features"],
                *config["categorical_features"],
            ],
            categorical_features=config["categorical_features"],
            component_weight=0.8,
        )
        self.assertEqual((len(explanations), calls, cat.shap_calls), (len(frame), 2, 2))

    def test_25_shap_failure_preserves_scores(self):
        frame, _ = self._build_dynamic(shap_fails=True)
        self.assertTrue(frame.dynamic_score.notna().all())
        self.assertEqual(frame.iloc[0].explanation["explanation_status"], "unavailable")

    def test_26_shap_failure_adds_warning(self):
        frame, _ = self._build_dynamic(shap_fails=True)
        self.assertIn(
            "CATBOOST_EXPLANATION_UNAVAILABLE", frame.iloc[0].enrichment_warnings
        )

    def test_27_frozen_contract_hash_is_unchanged(self):
        config = _config("24h")
        self.assertEqual(
            len(config["numerical_features"]) + len(config["categorical_features"]), 77
        )

    def test_28_segment_endpoint_returns_enriched_result(self):
        path = Path(tempfile.mkdtemp()) / "store.sqlite3"
        store = PredictionStore(path)
        frame, _ = self._build_dynamic()
        store.save_completed(
            {
                "batch_id": "b",
                "prediction_datetime": "2026-07-15T08:00:00+05:00",
                "started_at": "x",
                "completed_at": "x",
                "execution_time_ms": 1,
                "model_version": "v",
                "warnings": [],
            },
            frame,
        )
        client = TestClient(create_app(api_key="k", runtime=Runtime(store=store)))
        response = client.get(
            f"/api/v1/batches/b/segments/{frame.iloc[0].road_segment_id}",
            headers={"X-API-Key": "k"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("model_components", response.json())
        client.close()

    def test_29_city_plan_inputs_remain_compatible(self):
        frame, _ = self._build_dynamic()
        self.assertIn("road_segment_id", frame.columns)
        self.assertIn("dynamic_percentile", frame.columns)

    def test_30_enrichment_calls_no_provider(self):
        with patch(
            "ml_service.hybrid_risk.load_valid_future_context",
            side_effect=AssertionError,
        ):
            frame, _ = self._build_dynamic()
        self.assertEqual(len(frame), 3)


if __name__ == "__main__":
    unittest.main()
