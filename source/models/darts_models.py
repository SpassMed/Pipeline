import warnings
warnings.filterwarnings("ignore")  # avoid printing out absolute paths

from darts.models import NHiTSModel, TFTModel, RNNModel, NBEATSModel, TCNModel, TransformerModel ### Import additional models from DARTS Libarary
from darts.utils.likelihood_models import GaussianLikelihood

import torch.nn as nn

import pytorch_lightning as pl

# from utils import misc
# from darts.metrics.metrics import mse, dtw_metric, mape
# from optimization import evaluation_metrics

pl.seed_everything(42)

def dilate_loss(x, y):
    ## define alpha and gamma for dilate loss
    alpha=0.5 # default 0.4
    gamma=0.01 # default 0.001
    return dilate.dilate_wrapper(x,y, alpha=alpha, gamma=gamma)

def dilate_mse_loss(x, y): ### change this in future ### TODO: add MSE kernelized and 'alpha' for addition. 
    loss_mse_fn = nn.MSELoss()
    loss_mse = loss_mse_fn(x, y)
    loss_dilate = dilate_loss(x, y)
    return loss_mse + loss_dilate

def get_loss_function(loss_function: str):
    if loss_function == 'dilate':
        loss_fn = dilate_loss
    elif loss_function == 'mse':
        loss_fn = nn.MSELoss()
    elif loss_function is None:
        loss_fn = nn.MSELoss()
    elif loss_function == 'dilate_mse':
        loss_fn = dilate_mse_loss
    elif loss_function == 'mape':
        raise NotImplementedError
    return loss_fn

def build_nhits_model(
    training,
    validation,
    covariate_list: list,
    input_length: int,
    prediction_front: int,
    prediction_back: int,
    max_n_epochs: int,
    nr_epochs_val_period: int,
    batch_size: int,
    
    num_blocks: int,
    num_stacks: int,
    num_layers: int,
    layer_widths: int,
    activation: list[str],
    callbacks: list=None,
    model_name: str=None,
    gpu_num: list[int]=[0],
    lr: float=None, 
    loss_function: str=None,
    logger=None
    ):
    
    optimizer_kwargs = {'lr': lr}

    ## load loss function
    loss_fn = get_loss_function(loss_function=loss_function) # now it should work with any custom loss -- defined in get_loss_function!

    print(f'name of loss function: {loss_function}')
    
    model_nhits = NHiTSModel(
        input_chunk_length=int(input_length/5),
        output_chunk_length=int((prediction_front+prediction_back)/5),
        num_stacks=num_stacks,
        num_blocks=num_blocks,
        num_layers=num_layers,
        layer_widths=layer_widths,
        activation=activation,
        MaxPool1d = True,
        n_epochs=max_n_epochs,
        nr_epochs_val_period=nr_epochs_val_period,
        batch_size=batch_size,
        model_name=model_name,
        loss_fn=loss_fn,
        optimizer_kwargs= optimizer_kwargs,
        pl_trainer_kwargs={
            "accelerator": "gpu",
            "devices": gpu_num,
            # "auto_select_gpus": True,
            "callbacks": callbacks, 
            "logger":logger,
            },
        random_state=42,
        force_reset=True,
        save_checkpoints=True,
    )

    if covariate_list is not None:
        co_training, co_validation, _ = covariate_list
    else: co_training, co_validation, _ = None, None, None

    # train the model
    model_nhits.fit(
        series=training,
        val_series=validation,
        past_covariates=co_training,
        val_past_covariates=co_validation,            
        verbose=True)

    best_model_nhits =  model_nhits.load_from_checkpoint(model_name)

    return best_model_nhits


def build_tft_model(
    training,
    validation,
    covariate_list: list,
    input_length: int,
    prediction_front: int,
    prediction_back: int,
    max_n_epochs: int,
    nr_epochs_val_period: int,
    batch_size: int,
    num_hidden_size: int,
    num_lstm_layers: int,
    num_attention: int,
    num_dropout: float,
    callbacks: list=None,
    model_name: str=None,
    gpu_num: list[int]=[0],
    lr: float=None,
    loss_function: str=None,
    logger=None
    ):

    optimizer_kwargs = {'lr': lr}

    ## load loss function
    loss_fn = get_loss_function(loss_function=loss_function)

    ## Build TFT Model
    model_tft = TFTModel(
        input_chunk_length=int(input_length/5),
        output_chunk_length=int((prediction_front+prediction_back)/5),
        hidden_size=num_hidden_size,
        lstm_layers=num_lstm_layers,
        num_attention_heads=num_attention,
        batch_size=batch_size,
        dropout=num_dropout,
        n_epochs=max_n_epochs,
        nr_epochs_val_period=nr_epochs_val_period,
        add_relative_index=True,
        likelihood=None,
        model_name=model_name,
        loss_fn=loss_fn,
        optimizer_kwargs=optimizer_kwargs,
        pl_trainer_kwargs={
            "accelerator": "gpu",
            "devices": gpu_num,
            "callbacks": callbacks,
            "logger":logger
            },
        random_state=42,
        force_reset=True,
        save_checkpoints=True,
    )

    if covariate_list is not None:
        co_training, co_validation, co_testing = covariate_list
    else: co_training, co_validation, co_testing = None, None, None

    model_tft.fit(
        series=training, 
        val_series=validation, 
        past_covariates=co_training,
        val_past_covariates=co_validation,        
        verbose=True)

    best_model_tft =  model_tft.load_from_checkpoint(model_name)

    return best_model_tft

def build_tcn_model(
    training,
    validation,
    covariate_list: list,

    input_length: int,
    prediction_front: int,
    prediction_back: int,
    max_n_epochs: int,
    nr_epochs_val_period: int,
    
    batch_size: int,
    
    kernel_size: int = 3,
    num_filters: int = 3,
    num_layers: int = None,
    dilation_base: int =2,
    weight_norm: bool =False,
    dropout:int =0.2,

    callbacks: list=None,
    model_name: str=None,
    gpu_num: list[int]=[0],
    lr: float=None,
    loss_function: str=None,
    logger=None
    ):

    optimizer_kwargs = {'lr': lr}

    ## load loss function
    loss_fn = get_loss_function(loss_function=loss_function)    

    model_tcn = TCNModel(
        input_chunk_length=int(input_length/5),
        output_chunk_length=int((prediction_front+prediction_back)/5),
        kernel_size=kernel_size,
        num_filters=num_filters,
        dilation_base=dilation_base,
        weight_norm=weight_norm,
        num_layers=num_layers,
        dropout=dropout,
        n_epochs=max_n_epochs,
        nr_epochs_val_period=nr_epochs_val_period,
        batch_size=batch_size,
        model_name=model_name,
        loss_fn=loss_fn,
        optimizer_kwargs=optimizer_kwargs,        
        pl_trainer_kwargs={
            "accelerator": "gpu",
            "devices": gpu_num,
            "callbacks": callbacks, 
            "logger":logger,
            },
        force_reset=True,
        save_checkpoints=True,
    )

    if covariate_list is not None:
        co_training, co_validation, co_testing = covariate_list
    else: co_training, co_validation, co_testing = None, None, None    

    # train the model
    model_tcn.fit(
        series=training,
        val_series=validation,
        past_covariates=co_training,
        val_past_covariates=co_validation,                
        verbose=True)

    best_model_tcn =  model_tcn.load_from_checkpoint(model_name)

    return best_model_tcn


def build_nbeats_model(
    training,
    validation,
    covariate_list: list,

    input_length: int,
    prediction_front: int,
    prediction_back: int,
    max_n_epochs: int,
    nr_epochs_val_period: int,
    batch_size: int,
    
    num_blocks: int,
    num_layers: int,
    layer_widths: int,

    num_stacks: int = 2,
    activation: list[str] = None,
    callbacks: list=None,
    model_name: str = None,
    dropout: int = 0.2,
    generic_architecture: bool = True,
    gpu_num: list[int] = [0],
    expansion_coefficient_dim: int = 5, 
    trend_polynomial_degree: int = 2,
    lr: float=None,
    loss_function: str=None,    
    logger=None
    ):

    optimizer_kwargs = {'lr': lr}

    ## load loss function
    loss_fn = get_loss_function(loss_function=loss_function)

    # raise NotImplementedError
    model_nbeats = NBEATSModel(
        input_chunk_length=int(input_length/5),
        output_chunk_length=int((prediction_front+prediction_back)/5),
        num_stacks=num_stacks,
        num_blocks=num_blocks,
        num_layers=num_layers,
        generic_architecture=generic_architecture,
        dropout=dropout,
        layer_widths=layer_widths,
        activation=activation,
        n_epochs=max_n_epochs,
        nr_epochs_val_period=nr_epochs_val_period,
        batch_size=batch_size,
        model_name=model_name,
        expansion_coefficient_dim=expansion_coefficient_dim,
        trend_polynomial_degree=trend_polynomial_degree,
        loss_fn=loss_fn,
        optimizer_kwargs=optimizer_kwargs,
        pl_trainer_kwargs={
            "accelerator": "gpu",
            "devices": gpu_num,
            "callbacks": callbacks, 
            "logger":logger,
            },
        force_reset=True,
        save_checkpoints=True,
    )

    if covariate_list is not None:
        co_training, co_validation, co_testing = covariate_list
    else: co_training, co_validation, co_testing = None, None, None
    
    # train the model
    model_nbeats.fit(
        series=training,
        val_series=validation,
        past_covariates=co_training,
        val_past_covariates=co_validation,
        verbose=True)

    best_model_nbeats =  model_nbeats.load_from_checkpoint(model_name)

    return best_model_nbeats


def build_nhits_model_dilateloss( ## TODO: Fix this! 
    training,
    validation,
    covariate_list: list,
    input_length: int,
    prediction_front: int,
    prediction_back: int,
    max_n_epochs: int,
    nr_epochs_val_period: int,
    batch_size: int,
    num_blocks: int,
    num_stacks: int,
    num_layers: int,
    layer_widths: int,
    activation: list[str],
    callbacks: list=None,
    model_name: str = None,
    lr: float=None,
    loss_function: str=None,
    gpu_num: list[int] = [0], 
    logger=None
    ):

    optimizer_kwargs = {'lr': lr}

    ## load loss function
    loss_fn = get_loss_function(loss_function=loss_function)    

    model_nhits = NHiTSModel(
        input_chunk_length=int(input_length/5),
        output_chunk_length=int((prediction_front+prediction_back)/5),
        num_stacks=num_stacks,
        num_blocks=num_blocks,
        num_layers=num_layers,
        layer_widths=layer_widths,
        activation=activation,
        MaxPool1d = True,
        n_epochs=max_n_epochs,
        nr_epochs_val_period=nr_epochs_val_period,
        batch_size=batch_size,
        model_name=model_name,
        loss_fn=loss_fn,
        optimizer_kwargs=optimizer_kwargs,
        pl_trainer_kwargs={
            "accelerator": "gpu",
            "devices": gpu_num,
            "callbacks": callbacks, 
            "logger":logger,
            },
        force_reset=True,
        save_checkpoints=True,
    )

    if covariate_list is not None:
        co_training, co_validation, _ = covariate_list
    else: co_training, co_validation, _ = None, None, None

    # train the model
    model_nhits.fit(
        series=training,
        val_series=validation,
        past_covariates=co_training,
        val_past_covariates=co_validation,            
        verbose=True)

    best_model_nhits =  model_nhits.load_from_checkpoint(model_name)

    return best_model_nhits


def build_transformer_model(
    training,
    validation,
    covariate_list: list,

    input_length: int,
    prediction_front: int,
    prediction_back: int,
    max_n_epochs: int,
    nr_epochs_val_period: int,
    
    batch_size: int,
    
    d_model: int = 64,
    nhead: int = 4,
    num_encoder_layers: int = 3,
    num_decoder_layers: int = 3,
    dim_feedforward: int = 512,
    dropout: int = 0.1,
    callbacks: list=None,
    model_name: str=None,
    gpu_num: list[int]=[0],
    logger=None
    ):

    # torch.manual_seed(42)

    model_transformer = TransformerModel(
        input_chunk_length=int(input_length/5),
        output_chunk_length=int((prediction_front+prediction_back)/5),
        batch_size=batch_size,
        n_epochs=max_n_epochs,
        nr_epochs_val_period=nr_epochs_val_period,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        model_name=model_name,
        pl_trainer_kwargs={
            "accelerator": "gpu",
            "devices": gpu_num,
            "callbacks": callbacks,
            "logger":logger
            },
        random_state=42,
        force_reset=True,
        save_checkpoints=True,
        # optimizer_kwargs = {'lr':0.001},
    )

    if covariate_list is not None:
        co_training, co_validation, co_testing = covariate_list
    else: co_training, co_validation, co_testing = None, None, None    

    model_transformer.fit(
        series=training, 
        val_series=validation,
        past_covariates=co_training,
        val_past_covariates=co_validation,                         
        verbose=True)

    best_model_transformer =  model_transformer.load_from_checkpoint(model_name)

    return best_model_transformer

def build_deepar_model(
    training,
    validation,
    covariate_list: list,
    input_length: int,
    prediction_front: int,
    prediction_back: int,
    max_n_epochs: int,
    nr_epochs_val_period: int,
    batch_size: int,

    hidden_dim: int,
    n_rnn_layers: int,    
    num_dropout: float,

    callbacks: list=None,
    model_name: str=None,
    gpu_num: list[int]=[0],
    logger=None
    ):

    model_deepar = RNNModel(
        input_chunk_length= int(input_length/5), ## 72
        # output_chunk_length=int((prediction_front+prediction_back)/5), output_chunk_lengh is always 1.
        training_length=int(input_length/5), ## TODO: figure out the value for this!
        hidden_dim=hidden_dim,
        n_rnn_layers=n_rnn_layers,
        dropout=num_dropout,
        batch_size=batch_size,
        n_epochs=max_n_epochs,
        optimizer_kwargs={"lr": 1e-3},
        model_name=model_name,
        # likelihood=GaussianLikelihood(),
        model="LSTM",
        pl_trainer_kwargs={
            "accelerator": "gpu",
            "devices": gpu_num,
            "callbacks": callbacks,
            "logger":logger
            },
        random_state=42,
        force_reset=True,
        save_checkpoints=True,
    )

    if covariate_list is not None:
        co_training, co_validation, co_testing = covariate_list
    else: co_training, co_validation, co_testing = None, None, None

    model_deepar.fit(
        series=training, 
        val_series=validation,
        past_covariates=co_training,
        val_past_covariates=co_validation,        
        verbose=True)
    
    best_model_deepar = model_deepar.load_from_checkpoint(model_name)    

    return best_model_deepar


################ LOAD MODELS ################# TODO: FIX THIS!

def load_nhits_model(
    num_blocks: int,
    input_length: int,
    prediction_front: int,
    prediction_back: int,
    max_n_epochs: int,
    nr_epochs_val_period: int,
    batch_size: int,    
    num_stacks: int,
    num_layers: int,
    layer_widths: int,
    activation: list[str],
    callbacks: list=None,
    model_name: str = None,
    gpu_num: list[int]=[0],
    logger=None,
    loss_function: str="mse",
    lr: float=None
):
    loss_fn = get_loss_function(loss_function=loss_function)    


    optimizer_kwargs = {'lr': lr}

    model_nhits = NHiTSModel(
        input_chunk_length=int(input_length/5),
        output_chunk_length=int((prediction_front+prediction_back)/5),
        num_stacks=num_stacks,
        num_blocks=num_blocks,
        num_layers=num_layers,
        layer_widths=layer_widths,
        activation=activation,
        MaxPool1d = True,
        n_epochs=max_n_epochs,
        nr_epochs_val_period=nr_epochs_val_period,
        batch_size=batch_size,
        model_name=model_name,
        loss_fn = loss_fn,
        pl_trainer_kwargs={
            "accelerator": "gpu",
            "devices": gpu_num,
            # "auto_select_gpus": True,
            "callbacks": callbacks,
            "logger": logger,
            },
        random_state=42,
        optimizer_kwargs = optimizer_kwargs
        # force_reset=True, ## Donot reset trained model when loading from checkpoint.
        # save_checkpoints=True, ## Donot save the checkpoint when loading.
    )

    best_model_nhits = model_nhits.load_from_checkpoint(model_name)
    return best_model_nhits

def load_tft_model(
    input_length: int,
    prediction_front: int,
    prediction_back: int,
    max_n_epochs: int,
    nr_epochs_val_period: int,
    batch_size: int,
    num_hidden_size: int,
    num_lstm_layers: int,
    num_attention: int,
    num_dropout: float,
    callbacks: list=None,
    model_name: str=None,
    gpu_num: list[int]=[0],
    logger=None,
    loss_function: str="mse",
    lr: float=None
    ):
    loss_fn = get_loss_function(loss_function=loss_function)    
    optimizer_kwargs = {'lr': lr}

    ## Build TFT Model
    model_tft = TFTModel(
        input_chunk_length=int(input_length/5),
        output_chunk_length=int((prediction_front+prediction_back)/5),
        hidden_size=num_hidden_size,
        lstm_layers=num_lstm_layers,
        num_attention_heads=num_attention,
        batch_size=batch_size,
        dropout=num_dropout,
        n_epochs=max_n_epochs,
        nr_epochs_val_period=nr_epochs_val_period,
        add_relative_index=True,
        likelihood=None,
        loss_fn=loss_fn,
        model_name=model_name,
        pl_trainer_kwargs={
            "accelerator": "gpu",
            "devices": gpu_num,
            "callbacks": callbacks,
            "logger": logger,
            },
        random_state=42,
        optimizer_kwargs = optimizer_kwargs 
        # force_reset=True,
        # save_checkpoints=True,
        # optimizer_kwargs = {'lr':0.001},
    )

    best_model_tft =  model_tft.load_from_checkpoint(model_name)
    return best_model_tft

def load_tcn_model(
    input_length: int,
    prediction_front: int,
    prediction_back: int,
    max_n_epochs: int,
    nr_epochs_val_period: int,
    
    batch_size: int,
    
    kernel_size: int = 3,
    num_filters: int = 3,
    num_layers: int = None,
    dilation_base: int =2,
    weight_norm: bool =False,
    dropout:int =0.2,

    callbacks: list=None,
    model_name: str=None,
    gpu_num: list[int]=[0],
    logger=None,  
    loss_function: str="mse",
    lr: float=None
    ):
    loss_fn = get_loss_function(loss_function=loss_function)    
    optimizer_kwargs = {'lr': lr}

    model_tcn = TCNModel(
        input_chunk_length=int(input_length/5),
        output_chunk_length=int((prediction_front+prediction_back)/5),
        kernel_size=kernel_size,
        num_filters=num_filters,
        dilation_base=dilation_base,
        weight_norm=weight_norm,
        num_layers=num_layers,
        dropout=dropout,
        n_epochs=max_n_epochs,
        nr_epochs_val_period=nr_epochs_val_period,
        batch_size=batch_size,
        model_name=model_name,
        loss_fn = loss_fn,
        optimizer_kwargs = optimizer_kwargs,
        pl_trainer_kwargs={
            "accelerator": "gpu",
            "devices": gpu_num,
            "callbacks": callbacks, 
            "logger":logger,
            },
        # force_reset=True,
        # save_checkpoints=True,
    )

    best_model_tcn =  model_tcn.load_from_checkpoint(model_name)
    return best_model_tcn

def load_nbeats_model(
    input_length: int,
    prediction_front: int,
    prediction_back: int,
    max_n_epochs: int,
    nr_epochs_val_period: int,
    batch_size: int,
    
    num_blocks: int,
    num_stacks: int,
    num_layers: int,
    layer_widths: int,
    activation: list[str],
    callbacks: list=None,
    model_name: str = None,
    dropout: int = 0.2,
    generic_architecture: bool = True,
    gpu_num: list[int] = [0],
    expansion_coefficient_dim: int = 5, 
    trend_polynomial_degree: int = 2,
    logger=None,
    loss_function: str="mse",
    lr: float=None
    ):
    loss_fn = get_loss_function(loss_function=loss_function)    
    optimizer_kwargs = {'lr': lr}

    # raise NotImplementedError
    model_nbeats = NBEATSModel(
        input_chunk_length=int(input_length/5),
        output_chunk_length=int((prediction_front+prediction_back)/5),
        num_stacks=num_stacks,
        num_blocks=num_blocks,
        num_layers=num_layers,
        generic_architecture=generic_architecture,
        dropout=dropout,
        layer_widths=layer_widths,
        activation=activation,
        n_epochs=max_n_epochs,
        nr_epochs_val_period=nr_epochs_val_period,
        batch_size=batch_size,
        model_name=model_name,
        expansion_coefficient_dim=expansion_coefficient_dim,
        trend_polynomial_degree=trend_polynomial_degree,
        pl_trainer_kwargs={
            "accelerator": "gpu",
            "devices": gpu_num,
            "callbacks": callbacks, 
            "logger":logger,
            },
        random_state=42,
        loss_fn = loss_fn,
        optimizer_kwargs = optimizer_kwargs
        # force_reset=True,
        # save_checkpoints=True,
    )

    best_model_nbeats =  model_nbeats.load_from_checkpoint(model_name)

    return best_model_nbeats

def load_deepar_model(
    input_length: int,
    prediction_front: int,
    prediction_back: int,
    max_n_epochs: int,
    nr_epochs_val_period: int,
    batch_size: int,

    hidden_dim: int,
    n_rnn_layers: int,    
    num_dropout: float,

    callbacks: list=None,
    model_name: str=None,
    gpu_num: list[int]=[0],
    logger=None,
    loss_function: str="mse",
    lr: float=None
    ):

    loss_fn = get_loss_function(loss_function=loss_function)    
    optimizer_kwargs = {'lr': lr}

    model_deepar = RNNModel(
        input_chunk_length= int(input_length/5), ## 72
        # output_chunk_length=int((prediction_front+prediction_back)/5), output_chunk_lengh is always 1.
        training_length=int(input_length/5), ## TODO: figure out the value for this!
        hidden_dim=hidden_dim,
        n_rnn_layers=n_rnn_layers,
        dropout=num_dropout,
        batch_size=batch_size,
        n_epochs=max_n_epochs,
        model_name=model_name,
        # likelihood=GaussianLikelihood(),
        model="LSTM",
        pl_trainer_kwargs={
            "accelerator": "gpu",
            "devices": gpu_num,
            "callbacks": callbacks,
            "logger":logger
            },
        random_state=42,
        loss_fn = loss_fn,
        optimizer_kwargs={"lr": 1e-3},
        # force_reset=True,
        # save_checkpoints=True,
    )
    
    best_model_deepar = model_deepar.load_from_checkpoint(model_name)    

    return best_model_deepar

def load_transformer_model(
    input_length: int,
    prediction_front: int,
    prediction_back: int,
    max_n_epochs: int,
    nr_epochs_val_period: int,
    batch_size: int, 
    d_model: int = 64,
    nhead: int = 4,
    num_encoder_layers: int = 3,
    num_decoder_layers: int = 3,
    dim_feedforward: int = 512,
    dropout: int = 0.1,
    callbacks: list=None,
    model_name: str=None,
    gpu_num: list[int]=[0],
    logger=None,
    loss_function: str="mse",
    lr: float=None
    ):

    loss_fn = get_loss_function(loss_function=loss_function)    
    optimizer_kwargs = {'lr': lr}

    model_transformer = TransformerModel(
        input_chunk_length=int(input_length/5),
        output_chunk_length=int((prediction_front+prediction_back)/5),
        batch_size=batch_size,
        n_epochs=max_n_epochs,
        nr_epochs_val_period=nr_epochs_val_period,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        model_name=model_name,
        pl_trainer_kwargs={
            "accelerator": "gpu",
            "devices": gpu_num,
            "callbacks": callbacks,
            "logger":logger
            },
        random_state=42,
        loss_fn = loss_fn,
        optimizer_kwargs = optimizer_kwargs
        # force_reset=True,
        # save_checkpoints=True,
        # optimizer_kwargs = {'lr':0.001},
    )

    best_model_transformer =  model_transformer.load_from_checkpoint(model_name)

    return best_model_transformer


def build_dartsmodel(model_type, training, validation, covariate_list, model_hyper_params, model_training_params):

    if model_type == 'nhits':
        ## build and Train Model!
        mdl = build_nhits_model(training, validation, covariate_list, **model_hyper_params, **model_training_params)

    elif model_type == 'nhits_loss':
        mdl = build_nhits_model_dilateloss(training, validation, covariate_list, **model_hyper_params, **model_training_params) ## TODO: implement this properly
        nhits_customloss = True
        print(f"##### Is custom Loss Function turned ON? {nhits_customloss} ##### \n")

    elif model_type == 'tft':
        mdl = build_tft_model(training, validation, covariate_list, **model_hyper_params, **model_training_params)

    elif model_type == 'transformer':
        mdl = build_transformer_model(training, validation, covariate_list, **model_hyper_params, **model_training_params)

    elif model_type == 'nbeats':
        mdl = build_nbeats_model(training, validation, covariate_list, **model_hyper_params, **model_training_params)

    elif model_type == 'nbeats_RW':
        mdl = build_nbeats_model(training, validation, covariate_list, **model_hyper_params, **model_training_params)        

    elif model_type == 'nbeats_I':
        model_training_params['generic_architecture'] = False
        mdl = build_nbeats_model(training, validation, covariate_list, **model_hyper_params, **model_training_params)

    elif model_type == 'tcn':
        mdl = build_tcn_model(training, validation, covariate_list, **model_hyper_params, **model_training_params)

    elif model_type == 'deepar':
        mdl = build_deepar_model(training, validation, covariate_list, **model_hyper_params, **model_training_params)

    else: 
        print("model_type should be among: 'nhits', 'tft', 'transformer', 'nbeats', 'tcn', 'deepar' only!")
        raise NotImplementedError

    return mdl

## using it anywhere?
def load_dartsmodel(model_type, model_hyper_params, model_training_params):
    # Handel learning rate
    if model_type == 'nhits':
        mdl = load_nhits_model(**model_hyper_params, **model_training_params)

    elif model_type == 'tft':
        mdl = load_tft_model(**model_hyper_params, **model_training_params)

    elif model_type == 'transformer':
        mdl = load_transformer_model(**model_hyper_params, **model_training_params)

    elif model_type == 'nbeats':
        mdl = load_nbeats_model(**model_hyper_params, **model_training_params)

    elif model_type == 'tcn':
        mdl = load_tcn_model(**model_hyper_params, **model_training_params)

    elif model_type == 'deepar':
        mdl = load_deepar_model(**model_hyper_params, **model_training_params)

    else: 
        print("model_type should be among: 'nhits', 'nhits_loss', 'tft', 'transformer', 'nbeats', 'tcn', 'deepar' only!")
        raise NotImplementedError

    return mdl