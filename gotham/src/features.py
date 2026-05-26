from __future__ import annotations

import numpy as np
import pandas as pd


def add_gotham_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "pickup_datetime" in df.columns:
        dt = pd.to_datetime(df["pickup_datetime"], errors="coerce")

        df["pickup_hour"] = dt.dt.hour
        df["pickup_dayofweek"] = dt.dt.dayofweek
        df["pickup_month"] = dt.dt.month
        df["pickup_day"] = dt.dt.day
        df["is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)

        # Cyclic time encoding
        df["pickup_hour_sin"] = np.sin(2 * np.pi * df["pickup_hour"] / 24)
        df["pickup_hour_cos"] = np.cos(2 * np.pi * df["pickup_hour"] / 24)
        df["pickup_dayofweek_sin"] = np.sin(2 * np.pi * df["pickup_dayofweek"] / 7)
        df["pickup_dayofweek_cos"] = np.cos(2 * np.pi * df["pickup_dayofweek"] / 7)

        df = df.drop(columns=["pickup_datetime"])

    required = {"pickup_x", "pickup_y", "dropoff_x", "dropoff_y"}
    if required.issubset(df.columns):
        df["dx"] = df["dropoff_x"] - df["pickup_x"]
        df["dy"] = df["dropoff_y"] - df["pickup_y"]

        df["euclidean_distance"] = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2)
        df["manhattan_distance"] = df["dx"].abs() + df["dy"].abs()

        df["pickup_distance_from_origin"] = np.sqrt(df["pickup_x"] ** 2 + df["pickup_y"] ** 2)
        df["dropoff_distance_from_origin"] = np.sqrt(df["dropoff_x"] ** 2 + df["dropoff_y"] ** 2)

    return df