"""Stable train-only encoder used by Stage 19H experimental models."""

from __future__ import annotations

import pandas as pd


class TrainOnlyPreprocessor:
    def __init__(self, numeric: list[str], categorical: list[str]) -> None:
        self.numeric, self.categorical = numeric, categorical
        self.medians: dict[str, float] = {}
        self.codebooks: dict[str, dict[str, int]] = {}

    def fit(self, frame: pd.DataFrame) -> "TrainOnlyPreprocessor":
        self.medians = {
            column: float(pd.to_numeric(frame[column], errors="coerce").median())
            for column in self.numeric
        }
        self.codebooks = {}
        for column in self.categorical:
            values = frame[column].astype("string").fillna("__MISSING__")
            self.codebooks[column] = {
                str(value): index for index, value in enumerate(pd.unique(values))
            }
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=frame.index)
        for column in self.numeric:
            result[column] = (
                pd.to_numeric(frame[column], errors="coerce")
                .fillna(self.medians[column])
                .astype("float32")
            )
        for column in self.categorical:
            values = frame[column].astype("string").fillna("__MISSING__").astype(str)
            result[column] = (
                values.map(self.codebooks[column]).fillna(-1).astype("float32")
            )
        return result.loc[:, [*self.numeric, *self.categorical]]
