import os
import warnings
import time
warnings.filterwarnings("ignore")  # avoid printing out absolute paths
import argparse
import copy

import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.models import NHiTSModel ### Import additional models from DARTS Libarary
from darts.metrics.metrics import mse, dtw_metric, mape

import pickle
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

#### Hyperparameter Tuning ####
import optuna

from utils import misc #, train_model
from models import darts_models
from optimization.objectives import *
from optimization.evaluation_metrics import *

def check_torch_cuda():
    print(f"Torch version {torch.__version__}")
    print(f"Is Cuda Available: {torch.cuda.is_available()}")
    print(f"Number of Devices Available: {torch.cuda.device_count()}")
    print(f"Current Device selected: {torch.cuda.current_device()}")
    print(f"Name of current device: {torch.cuda.get_device_name(0)}")

    return torch.cuda.is_available()


def output_results(training_error_scores, validation_error_scores, testing_error_scores, column_name=None):

    if column_name is None:
        column_name = 'mean_errors'
        print(f"----- Results for All Target Columns Combined -----")
    else: 
        print(f"----- Results for Target Column: {column_name} -----")

    print(f"Training MSE Loss: {training_error_scores[column_name]['mse']} | \
           MAPE Loss: {training_error_scores[column_name]['mape']} | \
           DTW Loss: {training_error_scores[column_name]['dtw_metric']} \n")
    
    print(f"Validation MSE Loss: {validation_error_scores[column_name]['mse']} | \
           MAPE Loss: {validation_error_scores[column_name]['mape']} | \
           DTW Loss: {validation_error_scores[column_name]['dtw_metric']} \n")
    
    print(f"Testing MSE Loss: {testing_error_scores[column_name]['mse']} | \
          MAPE Loss: {testing_error_scores[column_name]['mape']} | \
          DTW Loss: {testing_error_scores[column_name]['dtw_metric']} \n")
    return


def create_model_directory(path):
    ## check path exits:
    isExists = os.path.exists(path)
    if not isExists:
        print(f'creating directory {path}')
        ## create directory
        os.makedirs(path)


def make_timeseries_dataset(
        train_dataset: pd.DataFrame, 
        val_dataset: pd.DataFrame,
        test_dataset: pd.DataFrame,
        time_col: str = 'time_idx',
        group_cols: list[str] = ['groups'],
        target_value: list[str] = ['meanbp']
):

    value_cols = target_value[0] if len(target_value)==1 else target_value

    ## Just transforming dataset to TimeSeries class format - compatible for DARTS package.
    training = TimeSeries.from_group_dataframe(
                                    txxt,
                                    time_col=time_col,
                                    group_cols=group_cols,  # individual time series are extracted by grouping `df` by `group_cols`
                                    value_cols=value_cols, # target variable
                                    )

    validation = TimeSeries.from_group_dataframe(
                                    val_dataset,
                                    time_col=time_col,
                                    group_cols=group_cols,  # individual time series are extracted by grouping `df` by `group_cols`
                                    value_cols=value_cols, # target variable
                                    )

    testing = TimeSeries.from_group_dataframe(
                                    test_dataset,
                                    time_col=time_col,
                                    group_cols=group_cols,  # individual time series are extracted by grouping `df` by `group_cols`
                                    value_cols=value_cols, # target variable
                                    )

    return training, validation, testing


def print_callback(study, trial):
    print(f"\n Current value: {trial.value}, Current params: {trial.params}")
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params} \n")


def build_training_dataset(experiment_config: dict):
    DATA_PATH = experiment_config['data_path'] ### Load from shared directory
    dataset_name = experiment_config['dataset_name']

    ## load pickle file
    load_path = os.path.join(DATA_PATH, 'forecasting', dataset_name)
    with open(load_path, 'rb') as file:
        all_data = pickle.load(file)
    f_data = all_data['original_data']

    # split the data in train_dataset, valid_dataset, test_dataset
    train_dataset, val_dataset, test_dataset = misc.prepare_train_test_set(f_data=f_data, extract_offset=False)
    TARGET_VALUE = experiment_config['target_value'] # ["meanbp_minmaxed_filter", "heartrate_minmaxed_filter"] # TODO: individual modalities as well?

    # if experiment_config['iscovariates']:
    covariate_value = experiment_config['covariate_value']

    co_training, co_validation, co_testing = make_timeseries_dataset(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        time_col='time_idx', # would always be static
        group_cols=['groups'], # would always be static
        target_value=covariate_value        
    )

    training, validation, testing = make_timeseries_dataset(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        time_col='time_idx', # would always be static
        group_cols=['groups'], # would always be static
        target_value=TARGET_VALUE
    )

    if experiment_config['iscovariates']:
        return training, validation, testing, co_training, co_validation, co_testing

    else:
        return training, validation, testing

def get_target_variates(args):

    minmax_columns = ["meanbp_minmaxed_filter", "heartrate_minmaxed_filter", "respiration_minmaxed_filter"]
    std_columns = ["meanbp_scaled_filter", "heartrate_scaled_filter", "respiration_scaled_filter"]
    raw_columns = ["meanbp", "heartrate", "respiration"]

    if args.target=='fm':
        target_value = ["meanbp_minmaxed_filter", "heartrate_minmaxed_filter"]
        covariate_value = [x for x in minmax_columns if x not in target_value]
        target_codename = "BPHR"

    elif args.target=='fs':
        target_value = ["meanbp_scaled_filter", "heartrate_scaled_filter"]
        covariate_value = [x for x in std_columns if x not in target_value]
        target_codename = "BPHR"
    
    elif args.target=='rw':
        target_value = ["meanbp", "heartrate"]
        covariate_value = [x for x in raw_columns if x not in target_value]
        target_codename = "BPHR"

    elif args.target=='fm_bp':
        target_value = ["meanbp_minmaxed_filter"]
        covariate_value = [x for x in minmax_columns if x not in target_value]
        target_codename = "BP"

    elif args.target=='fm_hr':
        target_value = ["heartrate_minmaxed_filter"]
        covariate_value = [x for x in minmax_columns if x not in target_value]
        target_codename = "HR"

    elif args.target=='fs_bp':
        target_value = ["meanbp_scaled_filter"]
        covariate_value = [x for x in std_columns if x not in target_value]
        target_codename = "BP"

    elif args.target=='fs_hr':
        target_value = ["heartrate_scaled_filter"]
        covariate_value = [x for x in std_columns if x not in target_value]
        target_codename = "HR"

    else: raise ValueError(f"{args.target} is not valid. please select from - fm: filter minmax, rw: raw, fs: filter std.")

    return target_value, covariate_value, target_codename