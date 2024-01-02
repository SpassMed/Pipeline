import os
import sys
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import random
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from typing import List, Tuple, Optional
from tqdm.auto import tqdm
from scipy import sparse
from timeit import default_timer as timer
import multiprocessing
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin

sys.path.append('/data/public/MLA/share/MLA_interns/Pipeline/source')

from sp1.features import (
    LaggedDiffFeatures,
    running_stats,
    lagged_fourier_transform,
)

features_vital = ["meanbp","heartrate","respiration"]
# 'meanbp_minmaxed_filter',
# 'heartrate_minmaxed_filter',
# 'respiration_minmaxed_filter']

features_vital_septic = [
'meanbp_minmaxed_filter',
'heartrate_minmaxed_filter']

def generate_random_date(start_date, end_date):
    # Convert input strings to datetime objects
    start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
    end_datetime = datetime.strptime(end_date, '%Y-%m-%d')

    # Calculate the time range
    time_range = end_datetime - start_datetime

    # Generate a random timedelta within the time range
    random_timedelta = timedelta(days=random.randint(0, time_range.days), hours=random.randint(0, 23), minutes=random.randint(0, 59), seconds=random.randint(0, 59))

    # Calculate the random date
    random_date = start_datetime + random_timedelta

    return random_date

class MissingValueCleaner(BaseEstimator, TransformerMixin):
    # The input to this class should be a list with missing values, the offset don't need to follow 5 minutes interval
    # This class is for cleaning missing values
    # As a result, the dataset will follow exactly 5 minutes interval
    # THINKING: we can just use an input list with 1 min interval, and then use this class to clean it
    def __init__(
            self,
            cutoff:int = 15,
            vitalsigns:List = [
                "systolicbp", "diastolicbp", "pp", "meanbp",
                "heartrate", "respiration", "spo2"]):

        self.cutoff = cutoff
        self.vitalsigns = vitalsigns

    def check_over(self, X):
        for vs in self.vitalsigns: 
            if len(X.loc[X["observationoffset"]%5 != 0, vs]) < len(X.loc[X["observationoffset"]%5 == 0, vs]) / 2:
                return True
            
            if X.loc[X["observationoffset"]%5 != 0, vs].isnull().sum() > self.cutoff:
                # Returns True if the number of missing values is over the cutoff
                return True
                        
        return False

    def carry_forward(self, X):
        # Imputation by carry forward
        X[self.vitalsigns] = X[self.vitalsigns].ffill(axis=0).bfill(axis=0)
        return X
    
    @classmethod
    def check_bp(cls, X):
        # Check unreasonable values and set them to NaN
        X.loc[
            (X["systolicbp"] <= X["diastolicbp"])
            | (X["systolicbp"] <= X["meanbp"]),
            ["systolicbp", "diastolicbp", "meanbp"]
        ] = np.NaN
        
        X.loc[
            (X["diastolicbp"] >= X["meanbp"]),
            ["diastolicbp", "meanbp"]
        ] = np.NaN
        
        X.loc[
            (X["systolicbp"] <= (X["diastolicbp"] + 4)),
            ["systolicbp", "diastolicbp"]
        ] = np.NaN
        
        # Imputation by carry forward
        X[["systolicbp", "diastolicbp", "meanbp"]] = X[["systolicbp", "diastolicbp", "meanbp"]].ffill(axis=0).bfill(axis=0)
        return X
        
    @classmethod
    def to_concat(cls, samples):
        datasets = []
        # Generating groups
        for idx, (vs, dx) in enumerate(tqdm(samples)):
            vs.insert(0, "groups", idx)
            datasets.append(vs.merge(
                dx, on="patientunitstayid"
            ))

        datasets = pd.concat(datasets).reset_index(drop=True)

        return datasets

    def fit(self, X, y=None):
        return self

    def transform(
            self,
            X :List[pd.DataFrame]
    )-> pd.DataFrame:
        samples = []
        for vs in tqdm(X):
            # if self.check_over(vs):
            #     continue
              
            vs = self.check_bp(self.carry_forward(vs))

            samples.append(
                vs.loc[
                    vs["observationoffset"]%5 == 0,
                    ["patientunitstayid", "observationoffset"]+self.vitalsigns
                ])

        return pd.DataFrame(pd.concat(samples).reset_index(drop=True))

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
        #'spo2': (50, 100),
        'meanbp': (0, 190),
        # 'systolicbp': (0, 300),
        # 'diastolicbp': (0, 200)
        #'pp': (40, 60)
    }

    feature_minmax_range = vital_minmax[col_name]

    scaler = CustomMinMaxScaler(feature_minmax_range[0], feature_minmax_range[1])

    assert isinstance(x, np.ndarray)

    final_signal = scaler.fit_transform(x.reshape(-1, 1))

    # assert len(final_signal) == main.END - main.BEGIN

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
        # print('for loop time:', end - start)

            # TODO: ADD rolling mean here to avoid filtereing. Done # there would be data leakage for 3 datapoints at max.

        col_scaled_filter = col_scaled + '_filter'

        start = timer()
        data[col_scaled_filter] = data[col_scaled].rolling(3).mean().fillna(method='ffill').fillna(method='bfill')
        end = timer()
        # print('fill-forward:', end - start)

    return data

def impute(VITALSIGNS, CUTOFF, vital_6hrs: pd.DataFrame):
    '''
    vital_6hrs: input 6 hours vital signs, should only have one group, one patient
    '''

    pipe = Pipeline(
        [
        ("MissingValueCleaner", MissingValueCleaner(
            cutoff=CUTOFF,
            vitalsigns=VITALSIGNS
        ))
        ]
    )

    samples = []
    for pid, group in tqdm(vital_6hrs.groupby("patientunitstayid")):
        samples.append(group)

    datasets = pipe.transform(samples)
    return datasets

def minmaxscaler(datasets: pd.DataFrame, col_names = ['meanbp', 'heartrate', 'respiration']):
    datasets["groups"] = 0
    final_datasets = minmaxscaler_groupwise(datasets, col_names)
    return final_datasets


# def load_data():
#     # dataset of vital signs
#     vital = pd.read_parquet("/home/lily/MLA/pipeline/5_patients.parquet.gzip", 
#     columns=["patientunitstayid", "observationoffset", "systolicbp", "diastolicbp", "pp", "heartrate", "respiration", "spo2"])
#     # vital["meanbp"] = vital["diastolicbp"] + 1/3 * (vital["systolicbp"] - vital["diastolicbp"])
#     return vital

def add_meanbp(vital):
    vital["meanbp"] = vital["diastolicbp"] + 1/3 * (vital["systolicbp"] - vital["diastolicbp"])
    return vital

def add_timeidx_groups(vital_6hrs):
    '''
    vital should contain ONLY 6 hours data
    '''
    vital_6hrs["time_idx"] = list(range(72))
    vital_6hrs["groups"] = 0
    return vital_6hrs


# feature generation
# lagged feature geenration; here LaggedDiffFeatures is used to generate lagged features
# LaggedDiffFeatures calculates difference between current and previous values

SAMPLING_RATE_MINUTES = 5 # 5 minutes
HOURS_FOR_INPUT_HR_sepsis = 6 # initial 6 hours of data for input
HOURS_FOR_INPUT_HR_septic = 9 # initial 9 hours of data for input

# feature generation
# lagged feature geenration; here LaggedDiffFeatures is used to generate lagged features
# LaggedDiffFeatures calculates difference between current and previous values


def extract_features(df):
    periods = []
    # using all periods
    periods = list(range(2, SAMPLING_RATE_MINUTES * HOURS_FOR_INPUT_HR_sepsis, 12))
    
    # return feature_extractor.transform(df)
    return pd.concat(
        objs=[
            running_stats(df, features_vital, periods=periods),
            LaggedDiffFeatures(df, features_vital, periods=periods),
            lagged_fourier_transform(df, features_vital, periods=periods),
        ],
        axis=1,
    )


def get_lagged_data(data): 
    # parallalize feature generation on grouped_by_patient

    num_cores = multiprocessing.cpu_count()
    print(f"Using {num_cores} cores")

    grouped_by_patient = data.sort_values(
        ["patientunitstayid"]
    ).groupby("groups")

    features = Parallel(n_jobs=num_cores)(
        delayed(extract_features)(df) for _, df in tqdm(grouped_by_patient)
    )
    features_df = pd.concat(features, axis=0)
    feature_names_generated = features_df.columns.tolist()

    # combining features to labels_df
    final_datasets = pd.concat([data, features_df], axis=1)
    return final_datasets,feature_names_generated


def extract_features_septic(df):
    periods = []
    # using all periods
    periods = list(range(2, SAMPLING_RATE_MINUTES * HOURS_FOR_INPUT_HR_septic, 12))
    
    # return feature_extractor.transform(df)
    return pd.concat(
        objs=[
            running_stats(df, features_vital_septic, periods=periods),
            LaggedDiffFeatures(df, features_vital_septic, periods=periods),
            lagged_fourier_transform(df, features_vital_septic, periods=periods),
        ],
        axis=1,
    )


def get_lagged_data_septic(data): 
    # parallalize feature generation on grouped_by_patient
    print("lagged before: ", data.shape)
    num_cores = multiprocessing.cpu_count()
    print(f"Using {num_cores} cores")

    grouped_by_patient = data.sort_values(
        ["patientunitstayid"]
    ).groupby("groups")

    features = Parallel(n_jobs=num_cores)(
        delayed(extract_features_septic)(df) for _, df in tqdm(grouped_by_patient)
    )
    # print( type(features[0]))
    features_df = pd.concat(features, axis=0)
    features_df = features_df.reset_index(drop=True)
    data = data.reset_index(drop=True)
    # print("Type: " + str(type(features_df)))
    feature_names_generated = features_df.columns.tolist()
    # print("lagged middle 1: ", data.shape)
    # print(data.head())
    # print("lagged middle 2: ", features_df.shape)
    # print(features_df)
    # combining features to labels_df
    # data = data.loc[~data.index.duplicated(keep='first')]
    # features_df = features_df.loc[~features_df.index.duplicated(keep='first')]
    final_datasets = pd.concat([data, features_df], axis=1)
    # print("lagged after: ", final_datasets.shape)
    # return data,feature_names_generated
    final_datasets["groups"] = 0
    return final_datasets,feature_names_generated


# if __name__ == "__main__":
#     vital = load_data()
    
#     VITALSIGNS = ["systolicbp", "diastolicbp", "pp", "meanbp", "heartrate", "respiration", "spo2"]
#     CUTOFF = 20

#     BEGIN = 0
#     END = 72
#     PATIENTID = 141233	
#     vital_6hrs = vital[(vital["patientunitstayid"] == PATIENTID)].iloc[BEGIN:END,:]

#     # impute missing values
#     datasets = impute(VITALSIGNS, CUTOFF, vital_6hrs)
#     # minmax scaler
#     final_datasets = minmaxscaler(datasets)
