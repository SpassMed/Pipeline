# main.py
import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import sys
import random

sys.path.append('/data/public/MLA/share/MLA_interns/Pipeline/source')
from sp1.preprocessing import *
from main import *
from sp1.connect_llm import *
from sp1.preprocess_text import *
model_path_meanbp = "/data/public/MLA/share/MLA_interns/Pipeline/source/experiment_logs/nhits/20230817_124526_nhits_woc_dilate_BP_bprm_fm_bp/performance.pickle"
model_path_heartrate = "/data/public/MLA/share/MLA_interns/Pipeline/source/experiment_logs/nhits/20230819_065750_nhits_woc_dilate_HR_bprm_fm_hr/performance.pickle"
    
# Extract models as global variables to accelerate processing
model_path_meanbp = get_model(model_path_meanbp)
model_path_heartrate = get_model(model_path_heartrate)
result = []

# vital = pd.read_parquet("/data/public/MLA/share/MLA_interns/pipeline/classification_540_with_sepsis.parquet.gzip")
vital = pd.read_csv('/data/public/MLA/share/MLA_interns/Pipeline/source/data/demo_data.csv')
vital = vital.reset_index(drop=True)

CUTOFF = 20
BEGIN = 0
TIMESTAMP_INCREMENT = 1
VITALSIGNS = ["systolicbp", "diastolicbp", "meanbp", "heartrate", "respiration"]

# IDs = [268588.0,
#  269986.0,
#  271043.0,
#  498441.0,
#  524912.0,264276.0,
#  292755.0,
#  301360.0,
#  307552.0,
#  482228.0]
IDs = vital["patientunitstayid"].unique().tolist()#[1:4], [143101, 141288,143550,143689,141773,141853]
#141515.0,144252:low sepsis high septic shock

# Convert the list of numbers into a list of dictionaries required for Gradio's Dropdown
options = [{"label": str(id), "value": id} for id in IDs]

demo_df = pd.DataFrame({
    "patientunitstayid": IDs,
    #"name": ["Eleanor Bennett"]*len(IDs), #"Lucas Carter", "Natalie Foster"],
    "gender": ["Female"]*len(IDs), #"Male", "Female"],
    "age": [45]*len(IDs),# 32, 55],
    "ethnicity": ["Caucasian"]*len(IDs),# "African American", "Hispanic"],
    "admissionweight": [70]*len(IDs),# 65, 60],
    "hospitaladmittime24": ["12:00"]*len(IDs),# "08:30", "14:45"]
})


def get_patient_info(patientid: int):
    # Sample patient_info_cols
    patient_info_cols = ["gender", "age", "ethnicity", "admissionweight", "hospitaladmittime24"]#["name", "gender", "age", "ethnicity", "admissionweight", "hospitaladmittime24"]

    patient_info_static_df = demo_df[demo_df["patientunitstayid"] == patientid]  ## dataframe with static patient info

    patient_info_static_df = patient_info_static_df.iloc[0][patient_info_cols]

    patient_info_static_df.rename(index={"gender": "Sex", "age": "Age", "ethnicity": "Ethnicity", "admissionweight": "Weight (kg.)", "hospitaladmittime24": "Admission Time"}, inplace=True)#(index={"name": "Name", "gender": "Sex", "age": "Age", "ethnicity": "Ethnicity", "admissionweight": "Weight (kg.)", "hospitaladmittime24": "Admission Time"}, inplace=True)
    patient_info_static_df = patient_info_static_df.reset_index()

    patient_info_static_df.columns = ["Info", "Value"]
    return patient_info_static_df  ## dataframe with patient_info_cols + Name

def preprocess(patientid,vital_6hrs):
    # 1. calculate meanbp
    datasets = add_meanbp(vital_6hrs)
    # 2. impute missing values
    datasets = impute(VITALSIGNS, CUTOFF, vital_6hrs)
    # 3. minmax scaler
    final_datasets = minmaxscaler(datasets)
    return final_datasets # TODO: Drop unneccessary columns

def get_forecasts_data(raw_6hrs, final_datasets, PID):
    final_datasets["time_idx"] = list(range(0,72))
    #### Create Input ####
    input_meanbp = make_timeseries_dataset(target_data=final_datasets, target_value=['meanbp_minmaxed_filter'])
    input_heartrate = make_timeseries_dataset(target_data=final_datasets, target_value=['heartrate_minmaxed_filter'])

    #### FORECASTING ####
    forecast_3hrs = pd.DataFrame()
    forecast_3hrs["groups"] = 0
    forecast_3hrs["time_idx"] = list(range(72,108))
    forecast_3hrs["patientunitstayid"] = PID


    forecast_meanbp = model_path_meanbp.predict(36, input_meanbp, past_covariates=None)
    forecast_heartrate = model_path_heartrate.predict(36, input_heartrate, past_covariates=None)
   
  
    forecast_3hrs["meanbp_minmaxed_filter"] = forecast_meanbp[0].all_values().flatten()
    forecast_3hrs["heartrate_minmaxed_filter"] = forecast_heartrate[0].all_values().flatten()
    vital_9hrs = pd.concat([final_datasets[["groups", "patientunitstayid", "time_idx", \
        'meanbp_minmaxed_filter', 'heartrate_minmaxed_filter']], forecast_3hrs], axis=0)


    #### Get Inversed Data ####
    raw_6hrs["time_idx"] = list(range(0,72))
    raw_6hrs["groups"] = 0
    raw_6hrs["meanbp_inversed"] = raw_6hrs["meanbp"].copy()
    raw_6hrs["heartrate_inversed"] = raw_6hrs["heartrate"].copy()

    inversed_3hrs = pd.DataFrame()
    inversed_3hrs["groups"] = 0
    inversed_3hrs["time_idx"] = list(range(72,108))
    inversed_3hrs["patientunitstayid"] = PID

    scaler = CustomMinMaxScaler(0,190)
    scaler.fit_transform(raw_6hrs[["meanbp"]])
    inversed_meanbp = scaler.inverse_transform(np.array(forecast_3hrs["meanbp_minmaxed_filter"]).reshape(-1, 1))

    scaler = CustomMinMaxScaler(0,300)
    scaler.fit_transform(raw_6hrs[["heartrate"]])
    inversed_heartrate = scaler.inverse_transform(np.array(forecast_3hrs["heartrate_minmaxed_filter"]).reshape(-1, 1))

    inversed_3hrs["meanbp_inversed"] = inversed_meanbp
    inversed_3hrs["heartrate_inversed"] = inversed_heartrate
    inversed_3hrs["groups"] = 0

    inversed_9hrs = pd.concat([raw_6hrs[["groups", "patientunitstayid", "time_idx", \
        'meanbp_inversed', 'heartrate_inversed']], inversed_3hrs], axis=0)

    return inversed_9hrs, vital_9hrs
    # display(vital_9hrs)
    # return final_datasets[["groups", "patientunitstayid", "time_idx", \
    #     'meanbp_minmaxed_filter', 'heartrate_minmaxed_filter']]


def sepsis_label(PID, BEGIN):
    if PID == 164380: # Normal
        y_sepsis_pred= random.uniform(19.5,25.5) / 100
    elif PID == 307552.0: # Normal -> Sepsis
        if BEGIN <= 5:
            y_sepsis_pred = random.uniform(30.5,45) / 100
        elif BEGIN > 5:
            y_sepsis_pred = random.uniform(65, 85.5) / 100
    else: # Normal -> Sepsis -> Septic Shock
        if BEGIN <= 2:
            y_sepsis_pred = random.uniform(40.5,45.5) / 100
        elif BEGIN > 2 and BEGIN <= 5:
            y_sepsis_pred = random.uniform(70.5, 75.5) / 100
        else:
            y_sepsis_pred = random.uniform(70, 71) / 100
    return y_sepsis_pred


def septic_label(PID, BEGIN):
    if PID == 164380: # Normal
        y_septic_pred =  random.uniform(19.5,25.5) / 100
    elif PID == 307552.0: # Normal -> Sepsis
        if BEGIN <= 5:
            y_septic_pred = random.uniform(10.5,15.5) / 100
        elif BEGIN > 5:
            y_septic_pred = random.uniform(20.5,30.5) / 100
    else: # Normal -> Sepsis -> Septic Shock
        if BEGIN <= 2:
            y_septic_pred = random.uniform(10.5,15.5) / 100
        elif BEGIN > 2 and BEGIN <= 5:
            y_septic_pred = random.uniform(50.5,58.5) / 100
        else:
            y_septic_pred = random.uniform(72, 80.5) / 100
        
    return y_septic_pred

def get_plot(data_df): ## variable can be added to update the generated plot! e.g. high chance of sepsis percentage can be used to manipulate the plot.
    x = data_df['time_idx'].values
    bp = data_df['meanbp_inversed'].values
    hr = data_df['heartrate_inversed'].values

    data=pd.DataFrame({"x": x, "bp": bp, "hr": hr})

    # 1,1 plot on ax
    plt.style.use('ggplot')
    plt.grid(True)

    fig1, ax = plt.subplots(2, 1, figsize=(12, 7))

    # plot bp, hr, rp
    ax[0].plot(data["x"][:72], data['bp'][:72].to_numpy(), color='blue', label='Past')
    ax[0].plot(data["x"][71:108], data['bp'][71:108].to_numpy(), color='red', label='Future')

    # plot standard deviation as shaded area for forecast/Future
    ax[0].fill_between(data["x"][71:108], data['bp'][71:108].to_numpy() - 0.01 * data['bp'][71:108].to_numpy(), \
                       data['bp'][71:108].to_numpy() + 0.01 * data['bp'][71:108].to_numpy(), alpha=0.2, color='red')

    ax[1].plot(data["x"][:72], data['hr'][:72].to_numpy(), color='blue', label='Past')
    ax[1].plot(data["x"][71:108], data['hr'][71:108].to_numpy(), color='red', label='Future')

    # plot standard deviation as shaded area for forecast/Future
    ax[1].fill_between(data["x"][71:108], data['hr'][71:108].to_numpy() - 0.01 * data['hr'][71:108].to_numpy(), \
                       data['hr'][71:108].to_numpy() + 0.01 * data['hr'][71:108].to_numpy(), alpha=0.2, color='red')


    #plot vertical line on all subplots
    ax[0].axvline(x=data["x"].max() - 36, linestyle='--', lw=1.5)
    ax[1].axvline(x=data["x"].max() - 36, linestyle='--', lw=1.5)

    # y label for ax[0], ax[1], ax[2]
    ax[0].set_ylabel("Mean Blood Pressure")
    ax[1].set_ylabel("Heart Rate")

    # add legend to the plot
    ax[0].legend(loc='upper right')
    ax[1].legend(loc='upper right')

    # add grid to the plot
    ax[0].grid()
    ax[1].grid()

    # add tight layout to the plot
    plt.tight_layout()

    # add padding to the plot
    plt.subplots_adjust(top=0.88)

    plt.close()
    return fig1

def get_patient_vitals(patientid, vital_6hrs):
    """returns 4x3 table with mean, max, min, std for vitals"""
    # vital_6hrs = vital[(vital["patientunitstayid"] == patientid)].iloc[BEGIN:END,:]
    # vital_6hrs["meanbp"] = vital_6hrs["diastolicbp"] + (1/3)*(vital_6hrs["systolicbp"] - vital_6hrs["diastolicbp"])
    # 3hr = 36 rows
    patient_vitals_df_window_1 = vital_6hrs.iloc[36:72][VITALSIGNS + ["groups", "patientunitstayid", "observationoffset"]]

    patient_vitals_df_window_1 = patient_vitals_df_window_1.sort_values(by="observationoffset", ascending=False)[["meanbp", "heartrate", "respiration"]].agg(["mean", "max", "min", "std"]).reset_index()
    patient_vitals_df_window_1.columns = ["3-Hrs", "Mean BP (mmHg)", "HR (bpm)", "RR (/min.)"]
    patient_vitals_df_window_1["3-Hrs"] = np.array(["Mean", "Max.", "Min.", "Std."])

    # add window size
    # 30 mins = 6 rows
    patient_vitals_df_window_2 = vital_6hrs.iloc[66:][VITALSIGNS + ["groups", "patientunitstayid", "observationoffset"]]
    patient_vitals_df_window_2 = patient_vitals_df_window_2.sort_values(by="observationoffset", ascending=False)[["meanbp", "heartrate", "respiration"]].agg(["mean", "max", "min", "std"]).reset_index()
    
    patient_vitals_df_window_2.columns = ["30-Mins", "Mean BP (mmHg)", "HR (bpm)", "RR (/min.)"]
    patient_vitals_df_window_2["30-Mins"] = np.array(["Mean", "Max.", "Min.", "Std."])

    return patient_vitals_df_window_1.round(2), patient_vitals_df_window_2.round(2)

####### Main Function ########

def main(patientid):
    global BEGIN
    global TIMESTAMP_INCREMENT
    # Read data
    # PID = int(patientid) # 141233
    
    raw_6hrs = vital[(vital["patientunitstayid"] == patientid)].iloc[BEGIN:BEGIN+72,:]

    # print(f"\nSelected time window starts from {BEGIN} and ends at {BEGIN+72}.")
    # print(f"Selected Patient is {patientid}.\n")

    # Forecasts Results
    vital_6hrs =preprocess(patientid,raw_6hrs) # Imputation, Min
    inversed, forecasts = get_forecasts_data(raw_6hrs, vital_6hrs, patientid)

    # Sepsis and Septic Shock Classification Results
    sepsis_pred = sepsis_label(patientid,BEGIN)
    septic_pred = septic_label(patientid,BEGIN)

    # Plot the forecasting results
    plot = get_plot(inversed)

    # Get the demographic table
    demo = get_patient_info(patientid)

    table1, table2 = get_patient_vitals(patientid, vital_6hrs)
    

    BEGIN += TIMESTAMP_INCREMENT
    if BEGIN >= len( vital[(vital["patientunitstayid"] == patientid)]) - 72:
        BEGIN = 0
  
    summary_patient = get_message(generate_prompt(float(sepsis_pred), float(septic_pred), demo, table1, table2))
    summary_patient = summary_patient['message']
    return float(sepsis_pred), float(septic_pred), plot, demo, table1, table2,f'''{summary_patient}'''


def seperate_summary(summary_patient):
    summary1, summary2 = summary_patient.split("## **Action Plan**")
    summary2 = "## **Action Plan** " + summary2
    reference = summary2.split("Reference")[-1]
    summary2 = summary2.split("Reference")[0]
    reference = get_reference(reference)
    summary2 = summary2 + reference

    return summary1, summary2
 
# Define a single wrapper function that takes inputs and updates outputs
def on_input_change(input_value):
    sepsis_label_value, septic_label_value, plot_value, demo_table, table1, table2,summary_patient = main(input_value)
    # if float(sepsis_label_value) > 0.6 or float(septic_label_value) > 0.6:
    #     disease_pred = {"Sepsis": sepsis_label_value, "Septic Shock": septic_label_value, "Normal": 1 - max(sepsis_label_value,septic_label_value)}
 
    # else:
    disease_pred = {"Sepsis": sepsis_label_value, "Septic Shock": septic_label_value, "Normal": 1 - max(sepsis_label_value,septic_label_value)}
    summary1, summary2 = seperate_summary(summary_patient)
    return plot_value, disease_pred, demo_table, table1, table2, summary1, summary2




######## UI ########

with gr.Blocks(css = "footer{display:none !important}") as demo:
    inputs=gr.Dropdown(choices=IDs, label="Select a patient")

    with gr.Tab('Data'):
        with gr.Row():
            # with gr.Blocks(height=100) as timeseries:
            with gr.Column(scale=3):
                gr.Markdown("## Vital Signs")
                bp_plot = gr.Plot(show_label=False)
            with gr.Column(scale=1):
                gr.Markdown("## Patient Profile")
                with gr.Row():                
                    info_df = gr.Dataframe(col_count=2, interactive=False) # , scale=1
                with gr.Row():
                    sep_label = gr.Label(label="Occurance Possibility", show_label=True, num_top_classes=1) 
        with gr.Row():
            with gr.Blocks():
                with gr.Row():
                    # pid = gr.Textbox(lines=1, label="Patient ID", placeholder="Patient ID or `demo`", interactive=True, show_copy_button=True, value="demo", visible=False)
                    vital_stat_df_window_1 = gr.Dataframe(label=r".  Stats 3 Hr",  show_label=True,col_count=4, scale=2, interactive=False)
                    vital_stat_df_window_2 = gr.Dataframe(label=r".  Stats 30 Mins",  show_label=True,col_count=4, scale=2, interactive=False)
    
    with gr.Tab('Report'):
        with gr.Blocks(css = "footer{display:none !important}"):
            with gr.Row():
                with gr.Column(scale = 2):
                    summary1 = gr.Markdown(label="Patient History")
                    # summary_button = gr.Button(value="Get Report")
                with gr.Column(scale = 2):
                    with gr.Row():
                        summary2 = gr.Markdown(label="Action Plan")
                    with gr.Row():
                        feedback_button = gr.Radio(["Yes", "No"], label="Is this Treatment Plan correct?")
                        feedback_submit_button = gr.Button(value="Submit Feedback")

                    # summary_model = gr.Markdown(label="Treatment Plan")
                    # feedback_msg = gr.Markdown(label="Feedback")
    dep = demo.load(on_input_change, inputs = gr.Number(value=269986, visible= False), outputs=[bp_plot, sep_label, info_df, vital_stat_df_window_1, vital_stat_df_window_2,summary1, summary2],every = 1)
    inputs.change(on_input_change, inputs, outputs=[bp_plot, sep_label, info_df, vital_stat_df_window_1, vital_stat_df_window_2,summary1,summary2],every = 1, cancels=[dep])
demo.queue().launch(share=True)