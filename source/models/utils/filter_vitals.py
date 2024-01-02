import pandas as pd
import numpy as np
import neurokit2 as nk
import os

import matplotlib.pyplot as plt

from configuration.config import DATA_PATH, DIR_PATH, VITALSIGNS

# Filter details from Marium
def filter_bp_signal(signal, sampling_rate=1/5, order=2, high_cut=0.05):
    signal_cleaned = nk.signal_filter(signal, highcut=high_cut, order=order, sampling_rate=sampling_rate) # define filter, use neurokit
    return signal_cleaned

# Filter details from Anubhav 
def filter_nbp_signal(signal, sampling_rate=1/5, order=4, high_cut=0.05):
    signal_cleaned = nk.signal_filter(signal, highcut=high_cut, order=order, sampling_rate=sampling_rate) # define filter, use neurokit

    return signal_cleaned

def filter_columns(column_names: str, data: pd.DataFrame) -> pd.DataFrame:

    # non_bp_cols = ['respiration', 'spo2']
    bp_cols = ["systolicbp", "diastolicbp", "meanbp", "pp", "heartrate"]

    for cols in column_names:

        if cols in bp_cols:
            apply_filter = filter_bp_signal

        else: apply_filter = filter_nbp_signal

        signal_1 = data.groupby(by='groups').apply(lambda x: apply_filter(x[cols]))

        col_filtered = cols + '_filtered'

        for patient in data['groups'].unique():
            signal_filtered = signal_1[int(patient)]
            data.loc[data['groups']==patient, col_filtered] = signal_filtered

    return data

if __name__ == '__main__':

    # read vital data
    # vitalPath = '/Users/anubhavbhatti/SpassMed/dataset/forecasting/021_forecasting_trainDataset_v0_1.parquet.gzip' # reading the new file # this file is now old.
    vitalPath = os.path.join(DATA_PATH, 'forecasting', '021_forecasting_trainDataset_v0_2.parquet.gzip')

    vital = pd.read_parquet(vitalPath)

    df_data = filter_columns(column_names=VITALSIGNS,
       data=vital)