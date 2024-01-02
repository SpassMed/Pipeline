import os
import numpy

# vitalFilteredPath = '/Users/anubhavbhatti/SpassMed/dataset/forecasting/021_forecasting_trainDataset_v0_2.parquet.gzip'


DIR_PATH = '/Users/anubhavbhatti/SpassMed'
DATA_PATH = '/Users/anubhavbhatti/SpassMed/dataset'

VITALSIGNS = ["systolicbp", "diastolicbp", "meanbp", "pp", "heartrate", "respiration", "spo2"]
DEFAULT_COL = ["patientunitstayid", "observationoffset"] + VITALSIGNS


