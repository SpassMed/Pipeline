import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def SMA(df: pd.DataFrame, col_names: list, periods: list):
    features = []
    for col in col_names:
        for period in periods:
            res = df[col].rolling(period).mean()
            res.rename(f"feature_sma_{col}_{period}", inplace=True)
            features.append(res)

    return pd.concat(features, axis=1)


def EMA(df: pd.DataFrame, col_names: list, periods: list):
    features = []
    for col in col_names:
        for period in periods:
            res = df[col].ewm(span=period).mean()
            res.rename(f"feature_ema_{col}_{period}", inplace=True)
            features.append(res)

    return pd.concat(features, axis=1)


def LaggedFeatures(df: pd.DataFrame, col_names: list, periods: list):
    features = []
    for col in col_names:
        for period in periods:
            res = df[col].shift(period)
            res.rename(f"feature_lagged_{col}_{period}", inplace=True)
            features.append(res)

    return pd.concat(features, axis=1)


def LaggedDiffFeatures(df: pd.DataFrame, col_names: list, periods: list):
    features = []
    for col in col_names:
        for period in periods:
            res = df[col].diff(period)
            res.rename(f"feature_lagged_diff_{col}_{period}", inplace=True)
            features.append(res)

    return pd.concat(features, axis=1)


# a function that returns the ratio between already calculated  ema and current value

def EMA_ratio(df: pd.DataFrame, col_names: list, periods: list):
    features = []
    for col in col_names:
        for period in periods:
            res = df[col].ewm(span=period).mean() / df[col]
            res.rename(f"feature_ema_ratio_{col}_{period}", inplace=True)
            features.append(res)

    return pd.concat(features, axis=1)

# a funtion that returns running volatility of heart rate and respiration


def volatility(df: pd.DataFrame, col_names: list, periods: list):
    features = []
    for col in col_names:
        for period in periods:
            res = df[col].rolling(period).std()
            res.rename(f"feature_volatility_{col}_{period}", inplace=True)
            features.append(res)

    return pd.concat(features, axis=1)


def ratio_between_cols(df: pd.DataFrame, col_names: list):
    """
    A function that returns the ratio between all possible pairs of columns
    """
    features = []
    for col1 in col_names:
        for col2 in col_names:
            if col1 != col2:
                res = df[col1] / df[col2]
                res.rename(f"feature_ratio_{col1}_{col2}", inplace=True)
                features.append(res)

    return pd.concat(features, axis=1)

# A function that returns the statistical features of the columns without look-ahead bias with running values in a window with kurtosis and skewness
def running_stats(df: pd.DataFrame, col_names: list, periods: list):
    features = []
    for col in col_names:
        for period in periods:
            res = df[col].rolling(period).agg(
                ["mean", "std", "min", "max", "median", "kurt", "skew"])
            res.rename(columns={"mean": f"feature_mean_{col}_{period}",
                                "std": f"feature_std_{col}_{period}",
                                "min": f"feature_min_{col}_{period}",
                                "max": f"feature_max_{col}_{period}",
                                "median": f"feature_median_{col}_{period}",
                                "kurt": f"feature_kurt_{col}_{period}",
                                "skew": f"feature_skew_{col}_{period}"}, inplace=True)
            features.append(res)

    return pd.concat(features, axis=1)

# A function that generates features based on fourier transform using scipy.fft.fft
def fourier_transform(df: pd.DataFrame, col_names: list, periods: list):
    features = []
    for col in col_names:
        for period in periods:
            res = df[col].rolling(period).apply(
                lambda x: np.abs(np.fft.fft(x))[1])
            res.rename(f"feature_fourier_{col}_{period}", inplace=True)
            features.append(res)

    return pd.concat(features, axis=1)

# a function to lag fourier transform features
def lagged_fourier_transform(df: pd.DataFrame, col_names: list, periods: list):
    features = []
    for col in col_names:
        for period in periods:
            res = df[col].rolling(period).apply(
                lambda x: np.abs(np.fft.fft(x))[1]).shift(period)
            res.rename(f"feature_lagged_fourier_{col}_{period}", inplace=True)
            features.append(res)

    return pd.concat(features, axis=1)


# Combine all features in a final transformer class which takes cols and periods as input

class FeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, col_names, periods):
        self.col_names = col_names
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        feats_arr = []
        # feats_arr.append(SMA(X, self.col_names, self.periods))
        # feats_arr.append(EMA(X, self.col_names, self.periods))
        feats_arr.append(LaggedFeatures(X, self.col_names, self.periods))
        # feats_arr.append(LaggedDiffFeatures(X, self.col_names, self.periods))
        # feats_arr.append(EMA_ratio(X, self.col_names, self.periods))
        feats_arr.append(ratio_between_cols(X, self.col_names))
        # feats_arr.append(volatility(
        #    X, ["heartrate", "respiration"], self.periods))

        return pd.concat(feats_arr, axis=1)
