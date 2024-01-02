'''
The file contains objective functions for all models used for hyperparameter optimization using Optuna
'''

import numpy as np

from optuna.integration import PyTorchLightningPruningCallback

import torch
import sys
sys.path.append('/data/public/MLA/share/MLA_interns/Pipeline')
from source.models import darts_models

from pytorch_lightning.callbacks import EarlyStopping

# from source.optimization.evaluation_metrics import *

def objective_nhits(trial, training, validation, covariate_list, model_name, gpu_num, experiment_config):

    pruner = PyTorchLightningPruningCallback(trial, monitor="val_loss")
    early_stopper = EarlyStopping(**experiment_config['early_stopping_kwargs'])
    callbacks = [pruner, early_stopper]

    optimize_lr = experiment_config.get("optimize_lr", "False")

    if optimize_lr:
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
        # optimizer_kwargs = {'lr': lr}
    else: lr = None

    num_blocks = trial.suggest_int("num_blocks", 2, 16)
    num_stacks = trial.suggest_int("num_stacks", 2, 32)
    num_layers = trial.suggest_int("num_layers", 2, 16)
    layer_widths = trial.suggest_categorical("layer_widths", [16, 32, 64, 128])
    activation = trial.suggest_categorical("activation", ['LeakyReLU', 'ReLU'])

    print(f"Current parameters for this Trial: {trial.params}")

    model_training_params = {
        'input_length': experiment_config['input_length'],
        'prediction_front':experiment_config['prediction_front'],
        'prediction_back':experiment_config['prediction_back'],
        'batch_size': experiment_config['batch_size'],
        'max_n_epochs': experiment_config['max_n_epochs'],
        'nr_epochs_val_period': experiment_config['nr_epochs_val_period']
    }

    model = darts_models.build_nhits_model( 
        training,
        validation,
        covariate_list,
        num_blocks=num_blocks,
        num_stacks=num_stacks,
        num_layers=num_layers,
        layer_widths=layer_widths,
        activation=activation,
        callbacks=callbacks,
        model_name=model_name,
        gpu_num=gpu_num,
        lr=lr,
        loss_function=experiment_config['loss_function'],
        **model_training_params
    )
    
    torch.cuda.empty_cache()

    # Evaluate how good it is on the validation set
    if covariate_list is not None:
        score = get_MSE_Scores(model, validation, covariate_list[1], model_training_params)

    else: 
        score = get_MSE_Scores(model, validation, covariate_list, model_training_params) # covariate_list is None here!

    return score if score != np.nan else float("inf")


def objective_nhits_customloss(trial, training, validation, covariate_list, model_name, gpu_num, experiment_config):

    pruner = PyTorchLightningPruningCallback(trial, monitor="val_loss")
    early_stopper = EarlyStopping(**experiment_config['early_stopping_kwargs'])
    callbacks = [pruner, early_stopper]

    optimize_lr = experiment_config.get("optimize_lr", "False")

    if optimize_lr:
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
        # optimizer_kwargs = {'lr': lr}
    else: lr = None

    num_blocks = trial.suggest_int("num_blocks", 2, 16)
    num_stacks = trial.suggest_int("num_stacks", 2, 32)
    num_layers = trial.suggest_int("num_layers", 2, 16)
    layer_widths = trial.suggest_categorical("layer_widths", [16, 32, 64, 128])
    activation = trial.suggest_categorical("activation", ['LeakyReLU', 'ReLU'])

    print(f"Current parameters for this Trial: {trial.params}")

    model_training_params = {
        'input_length': experiment_config['input_length'],
        'prediction_front':experiment_config['prediction_front'],
        'prediction_back':experiment_config['prediction_back'],
        'batch_size': experiment_config['batch_size'],
        'max_n_epochs': experiment_config['max_n_epochs'],
        'nr_epochs_val_period': experiment_config['nr_epochs_val_period']
    }

    model = darts_models.build_nhits_model_dilateloss( 
        training,
        validation,
        covariate_list,
        
        num_blocks=num_blocks,
        num_stacks=num_stacks,
        num_layers=num_layers,
        layer_widths=layer_widths,
        activation=activation,
        callbacks=callbacks,
        model_name=model_name,
        gpu_num=gpu_num,        
        lr=lr,
        loss_function=experiment_config['loss_function'],
        **model_training_params
    )
    
    torch.cuda.empty_cache()

    # Evaluate how good it is on the validation set
    if covariate_list is not None:
        # score = get_MSE_Scores(model, validation, covariate_list[1], model_training_params)
        score = get_DILATE_Scores(model, validation, covariate_list[1], model_training_params)
    else:
        # score = get_MSE_Scores(model, validation, covariate_list, model_training_params)
        score = get_DILATE_Scores(model, validation, covariate_list, model_training_params)

    return score if score != np.nan else float("inf")


def objective_nbeats(trial, training, validation, covariate_list, model_name, gpu_num, experiment_config):

    pruner = PyTorchLightningPruningCallback(trial, monitor="val_loss")
    early_stopper = EarlyStopping(**experiment_config['early_stopping_kwargs'])
    callbacks = [pruner, early_stopper]

    optimize_lr = experiment_config.get("optimize_lr", "False")

    if optimize_lr:
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
        # optimizer_kwargs = {'lr': lr}
    else: lr = None

    num_blocks = trial.suggest_int("num_blocks", 2, 20)
    num_stacks = trial.suggest_int("num_stacks", 8, 20)
    num_layers = trial.suggest_int("num_layers", 2, 20)
    layer_widths = trial.suggest_categorical("layer_widths", [32, 64, 128, 256])
    activation = trial.suggest_categorical("activation", ['SELU', 'LeakyReLU', 'ReLU'])
    # lr = trial.suggest_categorical('lr', [0.1, 0.01, 0.001, 0.0001, 0.00001])

    print(f"Current parameters for this Trial: {trial.params}")

    model_training_params = {
        'input_length': experiment_config['input_length'],
        'prediction_front':experiment_config['prediction_front'],
        'prediction_back':experiment_config['prediction_back'],
        'batch_size': experiment_config['batch_size'],
        'max_n_epochs': experiment_config['max_n_epochs'],
        'nr_epochs_val_period': experiment_config['nr_epochs_val_period'],
        'generic_architecture': experiment_config['generic_architecture']
    }

    # optimizer_kwargs = {'lr': lr}

    model = darts_models.build_nbeats_model(
        training,
        validation,
        covariate_list,
        num_blocks=num_blocks,
        num_stacks=num_stacks,
        num_layers=num_layers,
        layer_widths=layer_widths,
        activation=activation,
        callbacks=callbacks,
        model_name=model_name,
        gpu_num=gpu_num,
        lr=lr,
        loss_function=experiment_config['loss_function'],
        **model_training_params
    )
    
    torch.cuda.empty_cache()

    # Evaluate how good it is on the validation set
    if covariate_list is not None:
        score = get_MSE_Scores(model, validation, covariate_list[1], model_training_params)

    else: 
        score = get_MSE_Scores(model, validation, covariate_list, model_training_params) # covariate_list is None here!
    return score if score != np.nan else float("inf")


def objective_nbeats_I(trial, training, validation, covariate_list, model_name, gpu_num, experiment_config):

    pruner = PyTorchLightningPruningCallback(trial, monitor="val_loss")
    early_stopper = EarlyStopping(**experiment_config['early_stopping_kwargs'])
    callbacks = [pruner, early_stopper]

    optimize_lr = experiment_config.get("optimize_lr", "False")

    if optimize_lr:
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
        # optimizer_kwargs = {'lr': lr}
    else: lr = None

    num_blocks = trial.suggest_int("num_blocks", 4, 32)
    # num_stacks = trial.suggest_int("num_stacks", 8, 16)
    num_layers = trial.suggest_int("num_layers", 2, 32)
    layer_widths = trial.suggest_categorical("layer_widths", [64, 128, 256])
    activation = trial.suggest_categorical("activation", ['SELU', 'LeakyReLU', 'ReLU'])
    trend_polynomial_degree = trial.suggest_int("trend_polynomial_degree", 2, 8)
    # lr = trial.suggest_categorical('lr', [0.1, 0.01, 0.001, 0.0001, 0.00001])

    print(f"Current parameters for this Trial: {trial.params}")

    model_training_params = {
        'input_length': experiment_config['input_length'],
        'prediction_front':experiment_config['prediction_front'],
        'prediction_back':experiment_config['prediction_back'],
        'batch_size': experiment_config['batch_size'],
        'max_n_epochs': experiment_config['max_n_epochs'],
        'nr_epochs_val_period': experiment_config['nr_epochs_val_period'],
        'generic_architecture': experiment_config['generic_architecture']
    }

    model = darts_models.build_nbeats_model( #### TODO: FIX THIS!!
        training,
        validation,
        covariate_list,
        num_blocks=num_blocks,
        # num_stacks=num_stacks,
        num_layers=num_layers,
        layer_widths=layer_widths,
        activation=activation,
        trend_polynomial_degree=trend_polynomial_degree,
        callbacks=callbacks,
        model_name=model_name,
        gpu_num=gpu_num,
        lr=lr,        
        loss_function=experiment_config['loss_function'],
        **model_training_params
    )
    
    torch.cuda.empty_cache()

    # Evaluate how good it is on the validation set
    if covariate_list is not None:
        score = get_MSE_Scores(model, validation, covariate_list[1], model_training_params)

    else: 
        score = get_MSE_Scores(model, validation, covariate_list, model_training_params) # covariate_list is None here!
    return score if score != np.nan else float("inf")

def objective_tft(trial, training, validation, covariate_list, model_name, gpu_num, experiment_config):

    pruner = PyTorchLightningPruningCallback(trial, monitor="val_loss")
    early_stopper = EarlyStopping(**experiment_config['early_stopping_kwargs'])
    callbacks = [pruner, early_stopper]

    optimize_lr = experiment_config.get("optimize_lr", "False")

    if optimize_lr:
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
        # optimizer_kwargs = {'lr': lr}
    else: lr = None

    num_hidden_size = trial.suggest_int("num_hidden_size", 8, 128, 8)
    num_lstm_layers = trial.suggest_int("num_lstm_layers", 8, 24, 1)
    num_attention = trial.suggest_int("num_attention", 12, 32, 1)
    num_dropout = trial.suggest_categorical("num_dropout", [0.005, 0.01, 0.05])

    print(f"Current parameters for this Trial: {trial.params}")

    model_training_params = {
        'input_length': experiment_config['input_length'],
        'prediction_front':experiment_config['prediction_front'],
        'prediction_back':experiment_config['prediction_back'],
        'batch_size': experiment_config['batch_size'],
        'max_n_epochs': experiment_config['max_n_epochs'],
        'nr_epochs_val_period': experiment_config['nr_epochs_val_period']
    }

    model = darts_models.build_tft_model( ## TODO: FIX This parameters
        training, 
        validation,
        covariate_list,
        num_hidden_size=num_hidden_size,
        num_lstm_layers=num_lstm_layers,
        num_attention=num_attention,
        num_dropout=num_dropout,
        callbacks=callbacks,
        model_name=model_name,
        gpu_num=gpu_num,
        lr=lr,        
        loss_function=experiment_config['loss_function'],        
        **model_training_params 
    )

    torch.cuda.empty_cache()

    # Evaluate how good it is on the validation set
    if covariate_list is not None:
        score = get_MSE_Scores(model, validation, covariate_list[1], model_training_params)

    else: 
        score = get_MSE_Scores(model, validation, covariate_list, model_training_params) # covariate_list is None here!

    return score if score != np.nan else float("inf")


def objective_tcn(trial, training, validation, covariate_list, model_name, gpu_num, experiment_config):

    pruner = PyTorchLightningPruningCallback(trial, monitor="val_loss")
    early_stopper = EarlyStopping(**experiment_config['early_stopping_kwargs'])
    callbacks = [pruner, early_stopper]

    optimize_lr = experiment_config.get("optimize_lr", "False")

    if optimize_lr:
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
        # optimizer_kwargs = {'lr': lr}
    else: lr = None

    kernel_size = trial.suggest_int("kernel_size", 3, 10)
    num_filters = trial.suggest_int("num_filters", 3, 20)
    dilation_base = trial.suggest_int("dilation_base", 2, 10)
    weight_norm = trial.suggest_categorical("weight_norm", [True, False])


    print(f"Current parameters for this Trial: {trial.params}")

    model_training_params = {
        'input_length': experiment_config['input_length'],
        'prediction_front':experiment_config['prediction_front'],
        'prediction_back':experiment_config['prediction_back'],
        'batch_size': experiment_config['batch_size'],
        'max_n_epochs': experiment_config['max_n_epochs'],
        'nr_epochs_val_period': experiment_config['nr_epochs_val_period']
    }

    model = darts_models.build_tcn_model( #### TODO: FIX THIS!!
        training,
        validation,
        covariate_list,
        kernel_size=kernel_size,
        num_filters=num_filters,
        dilation_base=dilation_base,
        weight_norm=weight_norm,
        callbacks=callbacks,
        model_name=model_name,
        gpu_num=gpu_num,
        lr=lr,        
        loss_function=experiment_config['loss_function'],                
        **model_training_params
    )
    
    torch.cuda.empty_cache()

    # Evaluate how good it is on the validation set
    if covariate_list is not None:
        score = get_MSE_Scores(model, validation, covariate_list[1], model_training_params)

    else: 
        score = get_MSE_Scores(model, validation, covariate_list, model_training_params) # covariate_list is None here!

    return score if score != np.nan else float("inf")


def objective_deepar(trial, training, validation, covariate_list, model_name, gpu_num, experiment_config):

    pruner = PyTorchLightningPruningCallback(trial, monitor="val_loss")
    early_stopper = EarlyStopping(**experiment_config['early_stopping_kwargs'])
    callbacks = [pruner, early_stopper]

    optimize_lr = experiment_config.get("optimize_lr", "False")

    if optimize_lr:
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
        # optimizer_kwargs = {'lr': lr}
    else: lr = None

    hidden_dim = trial.suggest_int("hidden_dim", 8, 256, 8)
    n_rnn_layers = trial.suggest_int("n_rnn_layers", 8, 24, 1)
    num_dropout = trial.suggest_categorical("num_dropout", [0.01, 0.1, 0.5])

    print(f"Current parameters for this Trial: {trial.params}")

    model_training_params = {
        'input_length': experiment_config['input_length'],
        'prediction_front':experiment_config['prediction_front'],
        'prediction_back':experiment_config['prediction_back'],
        'batch_size': experiment_config['batch_size'],
        'max_n_epochs': experiment_config['max_n_epochs'],
        'nr_epochs_val_period': experiment_config['nr_epochs_val_period']
    }

    model = darts_models.build_deepar_model( ## TODO: FIX This parameters
        training, 
        validation,
        covariate_list,
        hidden_dim=hidden_dim,
        n_rnn_layers=n_rnn_layers,
        num_dropout=num_dropout,
        callbacks=callbacks,
        model_name=model_name,
        gpu_num=gpu_num,
        lr=lr,        
        loss_function=experiment_config['loss_function'],
        **model_training_params 
    )

    torch.cuda.empty_cache()

    # Evaluate how good it is on the validation set
    if covariate_list is not None:
        score = get_MSE_Scores(model, validation, covariate_list[1], model_training_params)

    else: 
        score = get_MSE_Scores(model, validation, covariate_list, model_training_params) # covariate_list is None here!

    return score if score != np.nan else float("inf")