from __future__ import annotations

import numpy as np
import pandas as pd


def add_gotham_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "pickup_datetime" in df.columns:
        dt = pd.to_datetime(df["pickup_datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce")

        if dt.isna().any():
            dt_fallback = pd.to_datetime(df["pickup_datetime"], format="%m/%d/%y %H:%M", errors="coerce")
            dt = dt.fillna(dt_fallback)

        if dt.isna().any():
            dt_fallback = pd.to_datetime(df["pickup_datetime"], errors="coerce")
            dt = dt.fillna(dt_fallback)

        df["pickup_hour"] = dt.dt.hour
        df["pickup_dayofweek"] = dt.dt.dayofweek
        df["pickup_month"] = dt.dt.month
        df["pickup_day"] = dt.dt.day
        df["is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)

        df["is_rush_hour"] = df["pickup_hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)
        df["is_late_night"] = df["pickup_hour"].isin([0, 1, 2, 3, 4]).astype(int)

        df["pickup_hour_sin"] = np.sin(2 * np.pi * df["pickup_hour"] / 24)
        df["pickup_hour_cos"] = np.cos(2 * np.pi * df["pickup_hour"] / 24)
        df["pickup_dayofweek_sin"] = np.sin(2 * np.pi * df["pickup_dayofweek"] / 7)
        df["pickup_dayofweek_cos"] = np.cos(2 * np.pi * df["pickup_dayofweek"] / 7)

        df = df.drop(columns=["pickup_datetime"])

    if "NumberOfPassengers" in df.columns:
        df["passenger_count_clipped"] = df["NumberOfPassengers"].clip(lower=0, upper=6)
        df["is_abnormal_passenger_count"] = (~df["NumberOfPassengers"].between(1, 6)).astype(int)

    required = {"pickup_x", "pickup_y", "dropoff_x", "dropoff_y"}
    if required.issubset(df.columns):
        df["dx"] = df["dropoff_x"] - df["pickup_x"]
        df["dy"] = df["dropoff_y"] - df["pickup_y"]

        df["abs_dx"] = df["dx"].abs()
        df["abs_dy"] = df["dy"].abs()

        df["euclidean_distance"] = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2)
        df["manhattan_distance"] = df["abs_dx"] + df["abs_dy"]
        df["distance_squared"] = df["euclidean_distance"] ** 2
        df["log_euclidean_distance"] = np.log1p(df["euclidean_distance"])

        df["direction_angle"] = np.arctan2(df["dy"], df["dx"])

        df["avg_x"] = (df["pickup_x"] + df["dropoff_x"]) / 2
        df["avg_y"] = (df["pickup_y"] + df["dropoff_y"]) / 2

        df["pickup_distance_from_origin"] = np.sqrt(df["pickup_x"] ** 2 + df["pickup_y"] ** 2)
        df["dropoff_distance_from_origin"] = np.sqrt(df["dropoff_x"] ** 2 + df["dropoff_y"] ** 2)

        # Coarse spatial grid features.
        # Floor division creates location buckets so tree models can learn zone-level traffic patterns.
        grid_size = 10.0
        df["pickup_zone_x"] = np.floor(df["pickup_x"] / grid_size)
        df["pickup_zone_y"] = np.floor(df["pickup_y"] / grid_size)
        df["dropoff_zone_x"] = np.floor(df["dropoff_x"] / grid_size)
        df["dropoff_zone_y"] = np.floor(df["dropoff_y"] / grid_size)

        # Numeric route-zone interaction.
        df["route_zone_hash"] = (
            df["pickup_zone_x"] * 1_000_000
            + df["pickup_zone_y"] * 10_000
            + df["dropoff_zone_x"] * 100
            + df["dropoff_zone_y"]
        )

    return df