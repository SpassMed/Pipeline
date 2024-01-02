import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, gc
import sys
from copy import deepcopy
import random

from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

import lightgbm as lgb
import xgboost as xgb
from scipy import sparse
import pickle
# import catboost as ctb

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.metrics import make_scorer
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV

# import optuna

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
sys.path.append('/home/daisy/MLA_test/classification/03_POC/cmkim/')

from features import (
    LaggedFeatures,
    LaggedDiffFeatures,
    running_stats,
    fourier_transform,
    lagged_fourier_transform,
)

import torch

# import shap

# set all random seeds
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

tqdm.pandas()
from timeit import default_timer as timer

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from typing import List

features_vital = [
    "meanbp",
    "heartrate",
    "respiration",
    'meanbp_minmaxed_filter',
    'heartrate_minmaxed_filter',
    'respiration_minmaxed_filter'
]

SAMPLING_RATE_MINUTES = 5 # 5 minutes
HOURS_FOR_INPUT_HR = 6 # initial 3 hours of data for input
HOURS_FOR_LABEL_HR = 3 # lastest 1 hour of data for label
PCT_FOR_QSOFA_POSITIVE = 0.3 # 50% of the data should be positive for QSOFA

periods = list(range(2, SAMPLING_RATE_MINUTES * HOURS_FOR_INPUT_HR, 12))


class CustomMinMaxScaler(MinMaxScaler, TransformerMixin, BaseEstimator):
    def __init__(self, data_min, data_max):
        super().__init__()

        self.data_min = data_min
        self.data_max = data_max
        self.FLOAT_DTYPES = (np.float64, np.float32, np.float16)

    def partial_fit_custom(self, X, y=None):
        """Online computation of min and max on X for later scaling.
        All of X is processed as a single batch. This is intended for cases
        when :meth:`fit` is not feasible due to very large number of
        `n_samples` or because X is read from a continuous stream.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        y : None
            Ignored.
        Returns
        -------
        self : object
            Fitted scaler.
        """
        # self._validate_params() ## Not implemented in baseestimater class

        feature_range = self.feature_range
        if feature_range[0] >= feature_range[1]:
            raise ValueError(
                "Minimum of desired feature range must be smaller than maximum. Got %s."
                % str(feature_range)
            )

        if sparse.issparse(X):
            raise TypeError(
                "MinMaxScaler does not support sparse input. "
                "Consider using MaxAbsScaler instead."
            )

        first_pass = not hasattr(self, "n_samples_seen_")
        X = self._validate_data(
            X,
            reset=first_pass,
            dtype=self.FLOAT_DTYPES,
            force_all_finite="allow-nan",
        )

        data_min = self.data_min
        data_max = self.data_max

        if first_pass:
            self.n_samples_seen_ = X.shape[0]
        else:
            data_min = np.minimum(self.data_min, data_min)
            data_max = np.maximum(self.data_max, data_max)
            self.n_samples_seen_ += X.shape[0]

        data_range = data_max - data_min
        self.scale_ = (feature_range[1] - feature_range[0]) / self._handle_zeros_in_scale(
            data_range, copy=True
        )
        self.min_ = feature_range[0] - data_min * self.scale_
        self.data_min_ = data_min
        self.data_max_ = data_max
        self.data_range_ = data_range
        return self

    def fit(self, X, y=None):
        """Compute the minimum and maximum to be used for later scaling.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the per-feature minimum and maximum
            used for later scaling along the features axis.
        y : None
            Ignored.
        Returns
        -------
        self : object
            Fitted scaler.
        """
        # Reset internal state before fitting
        self._reset()
        return self.partial_fit_custom(X, y)

    def _handle_zeros_in_scale(self, scale, copy=True, constant_mask=None):
        """Set scales of near constant features to 1.
        The goal is to avoid division by very small or zero values.
        Near constant features are detected automatically by identifying
        scales close to machine precision unless they are precomputed by
        the caller and passed with the `constant_mask` kwarg.
        Typically for standard scaling, the scales are the standard
        deviation while near constant features are better detected on the
        computed variances which are closer to machine precision by
        construction.
        """
        # if we are fitting on 1D arrays, scale might be a scalar
        if np.isscalar(scale):
            if scale == 0.0:
                scale = 1.0
            return scale
        elif isinstance(scale, np.ndarray):
            if constant_mask is None:
                # Detect near constant values to avoid dividing by a very small
                # value that could lead to surprising results and numerical
                # stability issues.
                constant_mask = scale < 10 * np.finfo(scale.dtype).eps

            if copy:
                # New array to avoid side-effects
                scale = scale.copy()
            scale[constant_mask] = 1.0
            return scale


def func_scaler(x, col_name):

### plausible values for vital signs

    vital_minmax = {
        'heartrate': (0, 300),
        'respiration': (0, 100),
        'meanbp': (0, 190)
    }

    feature_minmax_range = vital_minmax[col_name]

    scaler = CustomMinMaxScaler(feature_minmax_range[0], feature_minmax_range[1])

    assert isinstance(x, np.ndarray)

    final_signal = scaler.fit_transform(x.reshape(-1, 1))

    assert len(final_signal) == 108

    return final_signal

def minmaxscaler_groupwise(data: pd.DataFrame, col_names: [str, List[str]]):

    # col_names = ['systolicbp', 'diastolicbp', 'meanbp', 'pp', 'heartrate', 'respiration', 'spo2']

    for cols in col_names:
        # signal_1 = data.groupby(by='groups').apply(lambda x: scaler.fit_transform(np.array(x[cols]).reshape(-1, 1)))
        signal_1 = data.groupby(by='groups').apply(lambda x: func_scaler(np.array(x[cols]), cols))

        col_scaled = cols + '_minmaxed'

        start = timer()
        for patient in  tqdm(data['groups'].unique()):
            signal_scaled = signal_1[int(patient)]

            data.loc[data['groups']==patient, col_scaled] = signal_scaled
        end = timer()
        print('for loop time:', end - start)

            # TODO: ADD rolling mean here to avoid filtereing. Done # there would be data leakage for 3 datapoints at max.

        col_scaled_filter = col_scaled + '_filter'

        start = timer()
        data[col_scaled_filter] = data[col_scaled].rolling(3).mean().fillna(method='bfill').fillna(method='ffill')
        end = timer()
        print('fill-forward:', end - start)

    return data

# Sepsis-3 definition
def calculate_qsofa(row):
    """
    Calculate qsofa score for a given row
    """
    qsofa_score = 0
    if row["respiration"] >= 22:
        qsofa_score += 1
    if row["systolicbp"] < 100:
        qsofa_score += 1
    if row["gcs"] < 15:
        qsofa_score += 1
    return qsofa_score

def calculate_qsofa_label(row):
    """
    Calculate qsofa label for a given row
    """
    qsofa_score = calculate_qsofa(row)
    if qsofa_score >= 2:
        return 1
    else:
        return 0



def match_labels(labels_df,original_labels):
    input_df = labels_df.groupby("groups").head(HOURS_FOR_INPUT_HR * 60 // SAMPLING_RATE_MINUTES).copy()
    target_df = labels_df.groupby("groups").tail(HOURS_FOR_LABEL_HR * 60 // SAMPLING_RATE_MINUTES).copy()

    # our new label
    target_df.loc[:, "label_qsofa"] = target_df.progress_apply(calculate_qsofa_label, axis=1)
    new_labels = (target_df.groupby("groups")["label_qsofa"].mean() >= PCT_FOR_QSOFA_POSITIVE).astype(int).rename("new_labels").reset_index()

    original_labels = original_labels.merge(new_labels, on="groups", how="left")
    input_df = input_df.merge(new_labels, on="groups", how="left")

    exact_groups = input_df.groupby("groups").apply(
    lambda x: (x["label_binary"] == x["new_labels"]).mean()
    )
    # we only want data that: new_label and septic_label matches
    exact_groups = exact_groups[exact_groups == 1].index.tolist()
    input_df = input_df[input_df["groups"].isin(exact_groups)].copy()

    return input_df

def extract_features(df):
    # return feature_extractor.transform(df)

    return pd.concat(
        objs=[
            running_stats(df, features_vital, periods=periods),
            LaggedDiffFeatures(df, features_vital, periods=periods),
            lagged_fourier_transform(df, features_vital, periods=periods),
        ],
        axis=1,
    )


def get_lagged_features(input_df):
    # feature generation
    # lagged feature geenration; here LaggedDiffFeatures is used to generate lagged features
    # LaggedDiffFeatures calculates difference between current and previous values
    gc.collect()

    # parallalize feature generation on grouped_by_patient
    num_cores = multiprocessing.cpu_count()
    print(f"Using {num_cores} cores")


    grouped_by_patient = input_df.sort_values(
        ["patientunitstayid"]
    ).groupby("groups")

    features = Parallel(n_jobs=num_cores)(
        delayed(extract_features)(df) for _, df in tqdm(grouped_by_patient)
    )

    features_df = pd.concat(features, axis=0)
    feature_names_generated = features_df.columns.tolist()

    # combining features to labels_df
    input_df = pd.concat([input_df, features_df], axis=1)
    return input_df,feature_names_generated 

if __name__ == '__main__':
    # Define the data directory and file name
    DATA_DIR = "/home/daisy/MLA_dataset/"
    FILE_NAME = "sepsis_classification_trainDataset_9hrs.parquet.gzip"

    # Load the data
    labels_df = pd.read_parquet(os.path.join(DATA_DIR, FILE_NAME))
    
    #Convert septic label to 0 and 1
    labels_df["label_binary"] = labels_df["label"].apply(lambda x: 1 if x == "sepsis" else 0)

    #### Add min-max filter
    col_names = ['meanbp', 'heartrate', 'respiration']
    labels_df = minmaxscaler_groupwise(labels_df, col_names)

    ### Match labels and get lagged features
    original_labels = labels_df[["patientunitstayid", "groups", "label_binary"]].copy()
    input_df = match_labels(labels_df,original_labels)
    input_df,feature_names_generated = get_lagged_features(input_df)

    ### Train the model
    # Input the last row of each group
    filter_df = input_df.groupby("groups").tail(1).copy()
    # Input features:
    feature_names = features_vital + feature_names_generated

    # Model
    model_baseline_gbdt = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        max_depth=4,
        reg_alpha=10,
        reg_lambda=10,
        unbalance=True,
        subsample=0.2,
        colsample_bytree=0.2,
        n_estimators=400,
        num_leaves=30,
        n_jobs=-1,
        random_state=42,
    )

    # Sve the model
    model_baseline_gbdt.fit(filter_df[feature_names], filter_df['label_binary'])
    #filename = f"/home/daisy/MLA_Classifier/Classification/sepsis_classification_9hrs_LGBM.sav"
    #pickle.dump(model_baseline_gbdt, open(filename, 'wb'))