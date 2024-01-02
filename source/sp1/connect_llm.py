
import requests
import json
import numpy as np

response_message = ""

def call_spassmed_api(question):
    global response_message
    url = "http://arca.spass.dev:8001/report"
    
    headers = {
        "accept": "*/*",
        "Content-Type": "application/json"
    }

    payload = {
        "query":question,
        "max_tokens": 1000,
        "stream": False
    }


    json_data = {'data': json.dumps(payload)}
    print(requests)
    
    # response = requests.post(url, headers=headers, data=json_data)
    # print(response.text)
    # data = response.content.decode('utf-16')
    # print(data)

    try:
        response = requests.post(url, headers=headers, json=payload)
        # response.raise_for_status()

        # Check the content type
        content_type = response.headers.get('Content-Type', "application/json")
        print(content_type)
        
        if 'application/json' in content_type:
            jsonResponse = response.json()
            # Handle the JSON response as needed
            response_message = jsonResponse
            print("JSON Response:", jsonResponse)
        elif 'text/event-stream' in content_type:
            # Handle SSE response (if needed)
            sse_data = response.text
            print("SSE Response:", sse_data)
            response_message = sse_data

        else:
            print("Unexpected Content-Type:", content_type)

    except requests.exceptions.RequestException as e:
        print("Error: ", e)

# Example usage:
def get_message(question):
    call_spassmed_api(question)
    return response_message

def generate_prompt(sepsis_rate, septic_rate, patient_info,table1, table2):
    summary_patient = '''## **Profile**
    - **Gender:** Male
    - **Age:** 41
    - **Ethnicity:** Caucasian
    - **Location:** Intensive Care Unit

    ## **ICU Admission Reason**
    Admitted due to symptoms indicative of cardiac and respiratory disturbances.

    ## **Historical Medical Data**
    - **Cardiovascular Issues:** Recorded instances of ventricular disorders, congestive heart failure, cardiogenic shock, and cardiac arrest.
    - **Pulmonary Concerns:** Episodes of respiratory failure, characterized by hypoxemia and acute respiratory distress.
    
    ## **Vital Signs**
    - **Blood Pressure:** Demonstrated fluctuations, with means around 89-90 mmHg.
    - **Heart Rate:** Elevated rates averaging around 120-123 bpm, with highs reaching 127 bpm.
    - **Respiration:** Rates varied widely, from as low as 12/min to as high as 81/min.

    ## **Sepsis & Shock Metrics**
    - **Sepsis Risk:** 83% 
    - **Septic Shock Risk:** 63%

    ## **Additional Information**
    - **Allergies:** Make some context.
    - **Family History:** Make some context. Need details, especially regarding cardiovascular issues.
    - **Social Habits:** Make some context. Full assessment required to understand potential risk factors.
    '''
      
    format = summary_patient + \
            '''
            ## **Action Plan** 
            1. ** Subtitle1 ** 
            - First item 
            - Second item 

            2. ** Subtitle1 ** 
            - First item 

            ** Reference ** 
            - First item
            - Second item  
            '''
              
    out =  "The patient information is: " + str(patient_info) + \
        "; The 3 hour statistics are" + table1.to_string() +\
        "; The 30 minutes statistics are" + table2.to_string() + \
        "; The predicted rate of getting sepsis condition is: " + str(sepsis_rate) +\
        "; The predicted rate of getting septic shock is: " + str(septic_rate) +\
        "; Can you give me some action plan for doctor since the patient is now in icu?" +\
        "; can I have the response in this format: " + format 
    if float(sepsis_rate) <= 0.6 or float(septic_rate) <= 0.6:
        out += "Also, with the sepsis and septic prediction result, sepis and septic Shock are not likely to occur.'"
    return out

# Example:
# patient_info = "Name:  Eleanor Bennett, Sex: Female, Age: 45, Ethnicity: Caucasian, Weight (kg.): 70"
# sepsis_rate = 0.25
# septic_rate = 0.58
# q = generate_prompt(sepsis_rate, septic_rate, patient_info, "", "")
# print(get_message(q)['message'])


# print(get_message("How are you"))