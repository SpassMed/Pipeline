import argparse
import pandas as pd
import pickle
from collections import deque
import sys
import numpy as np

import sys
sys.path.append('/data/public/MLA/share/MLA_interns/pipeline/source')
from sp1.forecasting import *
from sp1.preprocessing import *
from sp1.features import *
from darts_logs import *
from IPython.display import display
# python3 main.py --patientid 141233 --CI_len 12

CUTOFF = 20
VITALSIGNS = ["systolicbp", "diastolicbp", "meanbp", "heartrate", "respiration"]

features_vital = [
"meanbp",
"heartrate",
"respiration",
'meanbp_minmaxed_filter',
'heartrate_minmaxed_filter',
'respiration_minmaxed_filter']

features_vital_septic = [
'meanbp_minmaxed_filter',
'heartrate_minmaxed_filter']


if __name__ == '__main__':
    # Input start time, end time, selected patientid, and CI length
    parser = argparse.ArgumentParser()

    parser.add_argument("--patientid", help="select a patient id", type=int) #141233
    parser.add_argument("--CI_len", help="select the length of CI", type=int) #12

    # Specify arguments
    args = parser.parse_args()
    PID = args.patientid
    CI_LEN =args.CI_len

    # Read data
    vital = pd.read_parquet("/data/public/MLA/share/MLA_interns/pipeline/5_patients.parquet.gzip")
    vital = vital.reset_index(drop=True)

    # Initialize a queue to store predictions
    predictions = deque([])
    # Initialize a queue to store predictions
    predictions_septic = deque([])

    for BEGIN in range(5):
        END = BEGIN + 72
        ### I. Preprocess data ###
        vital_6hrs = vital[(vital["patientunitstayid"] == PID)].iloc[BEGIN:END,:]

        print(f"\nSelected time window starts from {BEGIN} and ends at {END}.")
        print(f"Selected Patient is {PID}.\n")

        # 1. calculate meanbp
        datasets = add_meanbp(vital_6hrs)
        # 2. impute missing values
        datasets = impute(VITALSIGNS, CUTOFF, vital_6hrs)
        # 3. minmax scaler
        final_datasets = minmaxscaler(datasets)

        # 4. generate statistical features & lagged feateures
        final_datasets,feature_names_generated = get_lagged_data(datasets)

        # Input features
        feature_names = features_vital + feature_names_generated

        # display(final_datasets.head())
        # II. Classification: Sepsis
        filename_sepsis = f"/data/public/MLA/share/MLA_interns/pipeline/models/sepsis_classification_9hrs_LGBM.sav"
        sepsis_model = pickle.load(open(filename_sepsis, "rb"))
    
        X_test = final_datasets[feature_names].tail(1).copy()
        y_sepsis_pred = sepsis_model.predict_proba(X_test)[:,1]


        if len(predictions) < CI_LEN:
            predictions.append(y_sepsis_pred)
            print('Sepsis: Less than one hr, no prediction!!\nThe length of results is:', len(predictions))
        else:
            predictions.popleft()
            predictions.append(y_sepsis_pred)
            print('Predicted sepsis in one hour:\n',list(predictions))

            # predictions = np.asarray(predictions)
            if sum(predictions) >= int(CI_LEN * 0.7):
                print(f'\nThe likelihood of sepsis happening is {np.mean(np.asarray(predictions)) * 100}% \n Sepsis DETECTED')
            elif sum(predictions) < int(CI_LEN * 0.7):
                print(f'\nThe likelihood of sepsis happening is {np.mean(np.asarray(predictions)) * 100}% \n Sepsis NOT detected')
                
        # III. Forecasting vital signs 3 hrs ahead
        final_datasets["time_idx"] = list(range(0,72))
        #### Create Input ####
        input_meanbp = make_timeseries_dataset(target_data=final_datasets, target_value=['meanbp_minmaxed_filter'])
        input_heartrate = make_timeseries_dataset(target_data=final_datasets, target_value=['heartrate_minmaxed_filter'])
        
        #### FORECASTING ####
        forecast_3hrs = pd.DataFrame()
        forecast_3hrs["groups"] = 0
        forecast_3hrs["time_idx"] = list(range(72,108))
        forecast_3hrs["patientunitstayid"] = PID

        model_path_meanbp = "/data/public/MLA/share/MLA_interns/pipeline/source/experiment_logs/nhits/20230817_124526_nhits_woc_dilate_BP_bprm_fm_bp/performance.pickle"
        model_path_heartrate = "/data/public/MLA/share/MLA_interns/pipeline/source/experiment_logs/nhits/20230819_065750_nhits_woc_dilate_HR_bprm_fm_hr/performance.pickle"
        
        forecast_meanbp = get_forecasts(model_path_meanbp, input_meanbp)
        forecast_heartrate = get_forecasts(model_path_heartrate, input_heartrate)

        forecast_3hrs["meanbp_minmaxed_filter"] = forecast_meanbp[0].all_values().flatten()
        forecast_3hrs["heartrate_minmaxed_filter"] = forecast_heartrate[0].all_values().flatten()
        vital_9hrs = pd.concat([final_datasets[["groups", "patientunitstayid", "time_idx", \
            'meanbp_minmaxed_filter', 'heartrate_minmaxed_filter']], forecast_3hrs], axis=0)
        
        # display(vital_9hrs.head())

        # IV. Predict Septic Shock

        ### Preprocess data ###
        # generate statistical features & lagged feateures
        final_datasets_septic,feature_names_generated = get_lagged_data_septic(vital_9hrs)
        # display(final_datasets_septic.head())
        # Input features
        feature_names_septic = features_vital_septic + feature_names_generated
        # print(feature_names_septic)
        filename_septic = f"/data/public/MLA/share/MLA_interns/pipeline/models/septic_shock_classification_9hrs_LGBM.sav"
        septic_model = pickle.load(open(filename_septic, "rb"))
        X_test_septic = final_datasets_septic[feature_names_septic].tail(1).copy()
        y_septic_pred = septic_model.predict_proba(X_test_septic)[:,1]


        if len(predictions_septic) < CI_LEN:
            predictions_septic.append(y_septic_pred)
            print('Septic shock: Less than one hr, no prediction!!\nThe length of results is:', len(predictions_septic))
        else:
            predictions_septic.popleft()
            predictions_septic.append(y_sepsis_pred)
            print('Predicted septic shock in one hour:\n',list(predictions_septic))

            # predictions_septic = np.asarray(predictions_septic)
            if sum(predictions_septic) >= int(CI_LEN * 0.7):
                print(f'\nThe likelihood of septic shock happening is {np.mean(np.asarray(predictions_septic)) * 100}% \n Septic Shock DETECTED')
            elif sum(predictions_septic) < int(CI_LEN * 0.7):
                print(f'\nThe likelihood of septic shock happening is {np.mean(np.asarray(predictions_septic)) * 100}% \n Septic Shock NOT detected')
    


         



    

    
