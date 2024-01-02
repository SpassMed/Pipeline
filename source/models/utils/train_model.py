# implement training pipeline
# model, dataloaders, criterion, optimizer, num_epochs=25
import time
import copy

from utils.predicts_test import plot_predictions_1

from utils.misc import ExtractTimeSseriesBeforeDxOffset

import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from typing import Union
import torch
import torch.nn as nn

from models import base_model
import models

# from pytorch_forecasting.data import MultiNormalizer, TorchNormalizer, GroupNormalizer
from pytorch_forecasting.metrics import MAE, MAPE, MASE, RMSE, SMAPE, MultiHorizonMetric, MultiLoss
from pytorch_forecasting.data.timeseries import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import MultiNormalizer, TorchNormalizer, GroupNormalizer

import pytorch_lightning as pl

def get_in_out_size(dataloaders, model_type:str='mlp') -> Union[int, int]:
    x , y = next(iter(dataloaders))

    if model_type=='mlp:':
        input_size = x['encoder_cont'][0].shape[0]*x['encoder_cont'][0].shape[1]

    elif model_type=='cnn':
        # print(x['encoder_cont'][0].shape)
        input_size = x['encoder_cont'][0].shape[1] # B x T x C for pytorch

    elif model_type=='vgg':
        # print(x['encoder_cont'][0].shape)
        input_size = x['encoder_cont'][0].shape[1] # B x T x C for pytorch

    output_size = y[0][0].shape[0]

    return int(input_size), int(output_size)


def training_model(model_name, dataloaders, hidden_width=512, n_hidden_layers=3, hidden_dense_width=None,
                   criterion='RMSE', optimizer='adam', num_epochs=50, 
                   learning_rate=0.001, kernel_size:list=None, stride:list=None, padding:bool=False):

    since = time.time()

    val_acc_history, val_loss_history = [], []
    train_acc_history, train_loss_history = [], []

    # define and load model to memory
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # get input size and output size
    input_size, output_size = get_in_out_size(dataloaders=dataloaders['train'], model_type=model_name)
    # print('output size: ', output_size)

    if model_name == "mlp":
        model = base_model.ModelVanillaMLP(input_size=input_size, output_size=output_size, hidden_width=hidden_width, n_hidden_layers=n_hidden_layers)
        model = model.to(device)

    elif model_name=="cnn":
        model = models.cnn_v1.ModelConv(in_ch=input_size, out_ch=output_size, hidden_width=hidden_width, kernel_size=kernel_size, stride=stride, atype='relu', padding=padding)
        model = model.to(device)

    elif model_name=="vgg":
        isBN=True
        model = models.cnn_vgg.ModelConv_VGG(in_ch=input_size, out_ch=output_size, hidden_width=hidden_width, 
                                             hidden_dense_width=hidden_dense_width,
                                             kernel_size=kernel_size, 
                                             stride=stride, atype='relu', 
                                             isBN=isBN)
        model = model.to(device)

    else: model = base_model.ModelVanillaMLP().to(device) # Not Implement TODO
    
    best_model_wts = copy.deepcopy(model.state_dict())

    if optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    if criterion == 'RMSE':
        criterion = RMSE(reduction="mean")

    elif criterion == 'MAE':
        criterion = MAE()

    elif criterion == 'MAPE':
        criterion = MAPE()

    elif criterion == 'SMAPE':
        criterion = SMAPE()

    else: raise NotImplementedError

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')

    for epoch in range(0, num_epochs):

        if int(epoch % 5) == 0:
            print('Epoch {}/{}'.format(epoch + 1, num_epochs))
            print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                train_loss = []
            else:
                model.eval()   # Set model to evaluate mode
                val_loss = []

            veryhigh_Loss = 1e10

            # iterate over data
            for data, labels in dataloaders[phase]:
                inputs = data['encoder_cont'].permute(0, 2, 1) # B x C x T
                inputs = inputs.to(device,  dtype=torch.float)
                labels = labels[0].to(device,  dtype=torch.float)
                labels = labels.squeeze(2)
                # print("label", labels.size())

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    # print("output", outputs.size())
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    loss = float(loss.detach().cpu().numpy())
                    train_loss.append(loss)

                if phase == 'val':
                    loss = float(loss.detach().cpu().numpy())
                    val_loss.append(loss)

            # epoch_loss = loss_train_val  / inputs.size()[0]
            if phase == 'train':
                epoch_loss = np.stack(train_loss).mean(axis=0)

            elif phase == 'val':   
                epoch_loss = np.stack(val_loss).mean(axis=0)
                # scheduler.step(np.stack(val_loss).mean(axis=0))

            if int(epoch % 5) == 0:
                print('{} Loss: {:.8f}'.format(phase, epoch_loss))
            
            # deep copy the model
            if phase == 'val' and epoch_loss < veryhigh_Loss:
                veryhigh_Loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_loss)
                val_loss_history.append(epoch_loss)
            if phase == 'train':
                train_acc_history.append(epoch_loss) # dirty fix
                train_loss_history.append(epoch_loss)
                
        # decreasing learning rate with the epochs


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, val_loss_history, train_acc_history, train_loss_history


if __name__ == "__main__":

    DATA_PATH = '/home/bhatti/dataset'

    f_data = pd.read_parquet(
        # os.path.join(DATA_PATH, 'forecasting', "141_forecasting_VitalsFiltered_0.1.parquet.gzip") # Group-wise MinMax Scaling
        os.path.join(DATA_PATH, 'forecasting', "141_forecasting_VitalsFiltered_0.2.parquet.gzip") # Global MinMax Scaling.

    ) # update DATA_PATH according based on dataset directory.

    target_data = ExtractTimeSseriesBeforeDxOffset(extracted_samples=108).transform(f_data.copy())


        # Split dataset/groups in Train and Test dataset
    train_groups, test_groups = train_test_split(
        target_data["groups"].unique(), test_size=0.2, random_state=42
    ) # 80-20 split

    train_dataset = target_data[target_data["groups"].isin(train_groups)]
    test_val_dataset = target_data[target_data["groups"].isin(test_groups)]

    # create train and validation datasets
    test_groups, val_groups = train_test_split(
        test_val_dataset["groups"].unique(), test_size=0.5, random_state=42
    )

    test_dataset = test_val_dataset[test_val_dataset["groups"].isin(test_groups)]
    val_dataset = test_val_dataset[test_val_dataset["groups"].isin(val_groups)]

    # Reset group numbers in train_dataset and test_dataset # not sure if required.
    train_dataset["groups"]=[group for group in range(len(train_dataset["groups"].unique())) for _ in range(len(train_dataset["time_idx"].unique()))]
    val_dataset["groups"]=[group for group in range(len(val_dataset["groups"].unique())) for _ in range(len(val_dataset["time_idx"].unique()))]
    test_dataset["groups"]=[group for group in range(len(test_dataset["groups"].unique())) for _ in range(len(test_dataset["time_idx"].unique()))]

    INPUT_LENGTH=360 # 6hours
    PREDICTION_FRONT=180# 3hours?
    PREDICTION_BACK=0

    # TARGET_VALUE = ["meanbp_filtered", "heartrate_filtered", "respiration_filtered"]
    TARGET_VALUE = ["meanbp_filtered"]

    if len(TARGET_VALUE) == 1:
        normalizer = None
        
    else:
        normalizer = MultiNormalizer([TorchNormalizer(method="identity") for _ in range(len(TARGET_VALUE))])

    training = TimeSeriesDataSet(
        train_dataset,
        group_ids=["groups"],
        target=TARGET_VALUE[0] if len(TARGET_VALUE)==1 else TARGET_VALUE,
        time_idx="time_idx",
        max_encoder_length=int(INPUT_LENGTH/5),
        max_prediction_length=int((PREDICTION_FRONT+PREDICTION_BACK)/5),
        time_varying_unknown_reals=TARGET_VALUE + ['spo2_filtered', "heartrate_filtered", "respiration_filtered", 'systolicbp_filtered', 'diastolicbp_filtered', 'pp_filtered'], # , 
        # time_varying_known_reals=,
        target_normalizer = normalizer)

    validation = TimeSeriesDataSet.from_dataset(
        training, 
        val_dataset, 
        predict=True, 
        stop_randomization=True
    )

    testing = TimeSeriesDataSet.from_dataset(
        training, 
        test_dataset, 
        predict=True, 
        stop_randomization=True
    )

    # create dataloaders for model
    batch_size = 64  # set this between 32 to 128?
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    test_dataloader = testing.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    dataloaders_dict = {'train': train_dataloader, 'val': val_dataloader}

    model, _, _, _, _ = training_model('cnn', dataloaders_dict, hidden_width=[64, 256, 256], 
                                   kernel_size=[13, 7, 3], stride=[1, 1, 1], criterion='RMSE', 
                                   optimizer='adam', num_epochs=100, 
                                   learning_rate=0.0002, padding='same')

    plot_predictions_1(model, test_dataloader, 10, 'meanbp_filtered')

