import os
import warnings
warnings.filterwarnings("ignore")  # avoid printing out absolute paths

import pickle 
import numpy as np
import pandas as pd
from typing import List, Tuple, Union

from torch.utils.data import Dataset, DataLoader

class TimeSeriesEICU(Dataset):
    def __init__(self, dataset,
                 target,
                 covariates,
                 group_column,
                 time_idx_column,
                 target_name, 
                 history_length,
                 prediction_length) -> None:
        super(Dataset).__init__()

        self.data = dataset[group_column + time_idx_column + covariates]
        self.target = target [group_column + time_idx_column + target_name]
        self.history_length = history_length
        self.prediction_length = prediction_length

        self.group_num = dataset['groups'].unique()

        self.data_array = np.stack(self.data.groupby(by='groups').apply(lambda x: x.values).to_numpy(), axis=0)
        self.target_array = np.stack(self.target.groupby(by='groups').apply(lambda x: x.values).to_numpy(), axis=0)

    def __len__(self):
        return len(self.group_num)

    def __getitem__(self, index):
        return (self.data_array[index], self.target_array[index])


if __name__ == '__main__':

    DATA_PATH = '/home/bhatti/dataset'
    dataset_path = '141_forecasting_Vitals_v0.1.pickle'

    load_path = os.path.join(DATA_PATH, 'forecasting', dataset_path)

    ## load pickle file

    with open(load_path, 'rb') as file:
        all_data = pickle.load(file)


    all_data.keys()


    dataset = all_data['history_data']
    target = all_data['prediction_data']
    
    covariates = ['systolicbp_scaled_filter', 'diastolicbp_scaled_filter', 'meanbp_scaled_filter']
    group_column = ['groups']
    time_idx_column = ['time_idx']
    target_name = ['meanbp_scaled_filter']

    train_loader = TimeSeriesEICU(dataset, 
                                target, 
                                covariates=covariates,
                                group_column=group_column,
                                time_idx_column=time_idx_column,
                                target_name=target_name, 
                                history_length=72, 
                                prediction_length=36
                                )

    loader = DataLoader(train_loader, batch_size = 3, shuffle = False)

    for i, d in enumerate(loader):
        print(i, d[0].shape, d[1].shape)
        break