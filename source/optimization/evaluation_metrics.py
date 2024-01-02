'''
The File contains all the error metrics used for calculating errors scores at the end of training the models.

'''

import numpy as np
from darts.metrics.metrics import mse, dtw_metric, mape
from custom_loss import dilate

import torch

import logging, sys
# logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.CRITICAL)
log_me = logging.getLogger('test_logger')

def get_MSE(x, test_trues) -> float:
    return np.mean(mse(x, test_trues))

def get_MAPE(x, test_trues) -> float:
    return np.mean(mape(x, test_trues))

def get_DTW(x, test_trues) -> float:
    return np.mean(dtw_metric(x, test_trues))

def get_DILATE(x, test_trues) -> float:
    dilate_loss_score = dilate.dilate_wrapper(x, test_trues)

    log_me.warn(f'shape of dilate loss score: {dilate_loss_score.size()}')
    return torch.mean(dilate_loss_score)

def get_model_predictions(mdl, data, covariates, prediction_front, prediction_back):

    test_windows, test_trues, co_test_windows, _ = split_data_hist_forecat(data, covariates, split_point=72)
    forecast = mdl.predict(int((prediction_front+prediction_back)/5), test_windows, past_covariates=co_test_windows)

    return test_windows,test_trues,forecast

def get_DILATE_Scores(mdl, data, covariates, model_training_params):

    test_windows, test_trues, co_test_windows, _ = split_data_hist_forecat(data, covariates, split_point=72)
    x = mdl.predict(int((model_training_params['prediction_front']+model_training_params['prediction_back'])/5), test_windows, past_covariates=co_test_windows)

    ## why this? -- 'x' above is a list of Time-series objects. Dilate function accepts array shape [batch, time-steps, targets]
    x_array = [x1.values() for x1 in x]
    test_trues = [t.values() for t in test_trues]

    x_array = np.stack(x_array)
    test_trues = np.stack(test_trues) ## TODO: convert to torch tensor!

    test_trues = torch.from_numpy(test_trues)
    x_array = torch.from_numpy(x_array)

    log_me.warn(f'length of x: {x_array.shape} \n\n')
    log_me.warn(f'length of test_trues: {test_trues.shape}')
    return get_DILATE(x_array, test_trues)

def get_MSE_Scores(mdl, data, covariates, model_training_params):
    
    test_windows, test_trues, co_test_windows, _ = split_data_hist_forecat(data, covariates, split_point=72)
    x = mdl.predict(int((model_training_params['prediction_front']+model_training_params['prediction_back'])/5), test_windows, past_covariates=co_test_windows)

    return np.mean(mse(x, test_trues))

def get_errorscores(mdl, data, covariates, prediction_front, prediction_back) -> dict:
    
    results = {}
    test_windows, test_trues, forecast = get_model_predictions(mdl, data, covariates, prediction_front, prediction_back)

    results['mean_errors'] = {'mse': get_MSE(forecast, test_trues), 'mape': get_MAPE(forecast, test_trues), 'dtw_metric': get_DTW(forecast, test_trues)}

    target_columns = list(test_trues[0].columns)
        
    for cols in target_columns:
        forecast_colwise = [x[cols] for x in forecast]
        trues_colwise = [x[cols] for x in test_trues]
        results[cols] = {'mse': get_MSE(forecast_colwise, trues_colwise), 'mape': get_MAPE(forecast_colwise, trues_colwise), 'dtw_metric': get_DTW(forecast_colwise, trues_colwise)} 

    results['history'] = test_windows
    results['actual'] = test_trues
    results['forecast'] = forecast

    return results

def split_data_hist_forecat(data, covariates, split_point=72):
    test_windows = []
    test_trues = []

    co_test_windows = []
    co_test_trues = []

    for i in range(len(data)):
        test = data[i]
        test_window, test_true = test.split_before(split_point)
        test_windows.append(test_window)
        test_trues.append(test_true)

        if covariates is not None:
            test_covariate = covariates[i]
            co_test_window, co_test_true = test_covariate.split_before(split_point) #send covariates for training, validation, testing
            co_test_windows.append(co_test_window)
            co_test_trues.append(co_test_true) # not required! can act as future covariates but that'd be data leakage for this case.
        else: 
            co_test_windows = None
            co_test_trues = None

    return test_windows, test_trues, co_test_windows, co_test_trues