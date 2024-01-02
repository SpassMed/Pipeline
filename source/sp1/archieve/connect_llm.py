
import requests
import json

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

def generate_prompt(sepsis_rate, septic_rate, patient_info):
    out =  "My information is: " + str(patient_info) + \
        "; The predicted rate of getting sepsis condition is: " + str(sepsis_rate) +\
        "; The predicted rate of getting septic shock is: " + str(septic_rate) +\
        "; Can you give me some health advice?"
    return out

# Example:
patient_info = "Name:  Eleanor Bennett, Sex: Female, Age: 45, Ethnicity: Caucasian, Weight (kg.): 70"
sepsis_rate = 0.25
septic_rate = 0.58
q = generate_prompt(sepsis_rate, septic_rate, patient_info)
get_message(q)

print(get_message("How are you"))