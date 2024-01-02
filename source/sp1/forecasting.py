import os
import pickle
import os
import warnings
import time
warnings.filterwarnings("ignore")  # avoid printing out absolute paths
import argparse
import copy
import logging
import numpy as np
import pandas as pd
import sys
sys.path.append('/data/public/MLA/share/MLA_interns/Pipeline/source')

from matplotlib import pyplot as plt

plt.style.use('ggplot')
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

from models import darts_models
from optimization import objectives

def make_timeseries_dataset(target_data,
        time_col: str = 'time_idx',
        group_cols: list[str] = ['groups'],
        target_value: list[str] = ['meanbp_minmaxed_filter']
):  
    '''
    This function is to transform the dataset to TimeSeries class format - compatible for DARTS package.
    target_data: 6hrs vital signs data, must have columns: ['time_idx', 'groups', 'meanbp_minmaxed_filter']
    '''

    value_cols = target_value[0] if len(target_value)==1 else target_value
    
    ## Just transforming dataset to TimeSeries class format - compatible for DARTS package.
    dataset = TimeSeries.from_group_dataframe(
                                    target_data,
                                    time_col=time_col,
                                    group_cols=group_cols,  # individual time series are extracted by grouping `df` by `group_cols`
                                    value_cols=value_cols, # target variable
                                    )
    return dataset

def get_forecasts(path, input):
    
    
    trained_model_name = path.split("/")[-2]

    trained_model_performance = path

    # load performance file for loading model_parameters and config file

    with open(trained_model_performance, 'rb') as file:
        model_performance = pickle.load(file)

    wandb_logger = WandbLogger(name=f'{trained_model_name}_wblogger', project='nhits')

    ## load model from darts model directory with model_parameters and perform predict 
    model_best_parameters = model_performance['model_best_parameters']
    model_best_parameters['model_name'] = model_performance['model_name']
    model_best_parameters['gpu_num'] = 2
    model_best_parameters['logger'] = wandb_logger

    experiment_config = model_performance['experiment_config']
    model_training_parameters = model_performance['model_training_params']

    early_stopper = EarlyStopping(**experiment_config['early_stopping_kwargs']) # "val_loss", min_delta=0.001, patience=15, verbose=True
    callbacks = [early_stopper]
    model_best_parameters['callbacks'] = callbacks
    print(model_best_parameters)

    ## implement load_model method.
    loaded_model = darts_models.load_dartsmodel(model_type='nhits', 
                                                model_hyper_params=model_best_parameters, 
                                                model_training_params=model_training_parameters)
    


    forecast = loaded_model.predict(36, input, past_covariates=None)
    return forecast


def get_model(path):
    
    
    trained_model_name = path.split("/")[-2]

    trained_model_performance = path

    # load performance file for loading model_parameters and config file

    with open(trained_model_performance, 'rb') as file:
        model_performance = pickle.load(file)

    wandb_logger = WandbLogger(name=f'{trained_model_name}_wblogger', project='nhits')

    ## load model from darts model directory with model_parameters and perform predict 
    model_best_parameters = model_performance['model_best_parameters']
    model_best_parameters['model_name'] = model_performance['model_name']
    model_best_parameters['gpu_num'] = 2
    model_best_parameters['logger'] = wandb_logger

    experiment_config = model_performance['experiment_config']
    model_training_parameters = model_performance['model_training_params']

    early_stopper = EarlyStopping(**experiment_config['early_stopping_kwargs']) # "val_loss", min_delta=0.001, patience=15, verbose=True
    callbacks = [early_stopper]
    model_best_parameters['callbacks'] = callbacks
    print(model_best_parameters)

    ## implement load_model method.
    loaded_model = darts_models.load_dartsmodel(model_type='nhits', 
                                                model_hyper_params=model_best_parameters, 
                                                model_training_params=model_training_parameters)
    

    return loaded_model

# if __name__ == '__main__':

#     #### Preprocessing ####
#     vital = pd.read_parquet("/home/lily/MLA/pipeline/5_patients.parquet.gzip")
#     BEGIN = 0
#     END = 72
#     PATIENTID = 141233	
#     vital_6hrs = vital[(vital["patientunitstayid"] == PATIENTID)].iloc[BEGIN:END,:]
    
#     # impute missing values
#     datasets = impute(VITALSIGNS, CUTOFF, vital_6hrs)
#     # minmax scaler
#     final_datasets = minmaxscaler(datasets)

#     #### Create Input ####
#     input_meanbp = make_timeseries_dataset(target_data=final_datasets, target_value=['meanbp_minmaxed_filter'])
#     input_heartrate = make_timeseries_dataset(target_data=final_datasets, target_value=['heartrate_minmaxed_filter'])
    
#     #### FORECASTING ####
#     forecast_3hrs = pd.DataFrame()
#     forecast_3hrs["groups"] = 0
#     forecast_3hrs["time_idx"] = list(range(72,108))
#     forecast_3hrs["patientunitstayid"] = PATIENTID

#     model_path_meanbp = "/home/vivi/MLA/forecasting/03_POC/vivi/source/experiment_logs/nhits/\
#         20230817_124526_nhits_woc_dilate_BP_bprm_cpfm_bp/performance.pickle"
#     model_path_heartrate = "/home/vivi/MLA/forecasting/03_POC/vivi/source/experiment_logs/nhits/\
#         20230819_065750_nhits_woc_dilate_HR_bprm_fm_hr/performance.pickle"
    
#     forecast_meanbp = get_forecasts(model_path_meanbp, input_meanbp)
#     forecast_heartrate = get_forecasts(model_path_heartrate, input_heartrate)

#     forecast_3hrs["meanbp_minmaxed_filter"] = forecast_meanbp[0].all_values().flatten()
#     forecast_3hrs["heartrate_minmaxed_filter"] = forecast_heartrate[0].all_values().flatten()

#     vital_9hrs = pd.concat([vital_6hrs[["groups", "patientunitstayid", "time_idx", \
#         'meanbp_minmaxed_filter', 'heartrate_minmaxed_filter']], forecast_3hrs], axis=0)
#     print(vital_9hrs)

    