from openai import OpenAI
import argparse
from datetime import datetime
import os
import re
import ast
import pickle

def get_env(record, summary,folder_path):
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "Work as an assistant on this task.\n\nI have a dataset about car crashes, records are crash summaries and is in text format. It is the text description of the crash, it contains road networks, traffic participants, trajectories, and environmental information.\n\nNow, I need you to extract the:\n\n1. Environment information including  weather and time. For the weather, your answer must be within these values:[Sunny, Cloudy, Overcast, Rainy, Snowy, Foggy, Windy]. For the time, your answer must be within these two values: [Daytime, Nighttime]\n\n2. Car type from the dataset. Your answer must be within these two values: [Car, Truck]\n\nYour output must be structured in this way: {'Weather': '<your answer for  the weather>', 'Time': 'your answer for the time', 'Car type': '<your answer for the car type>'}\n\nNote: Your answer should be in quotation marks.\n\nI will first show you one example to help you better understand this task."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Summary:\nVehicle one (V1 - case vehicle), a 2002 Subaru Forester, 4-door utility vehicle was traveling west in the westbound lane of a two-lane, two-way roadway and was negotiating a right curve. Vehicle two (V2), a 2002 Buick Rendezvous, 4-door utility vehicle was traveling east in the eastbound lane of the same roadway and was negotiating a left curve. It was daylight, snowing, and the bituminous road was icy. The driver of V1 lost control on the icy road and began to rotate counterclockwise. V1 rotated approximately 90 degrees, crossed the centerline and entered the eastbound lane. The driver of V2 could not avoid V1 and the front of V2 struck the right side of V1 in a T-type configuration. Both vehicles were towed due to disabling vehicle damage. The 36-year-old male driver of V1 (case occupant) was using the available three-point seat belt but no airbags deployed in the driver's seating position. He was transported via ground ambulance to a regional level-one trauma center.\n\nAnalysis process:\n1. Environment information extraction\nFrom the Summary, we can first locate the sentence for describing weather and time: 'It was daylight, snowing, and the bituminous road was icy.'\nFrom the description, we can see that the answer for weather is sunny and the answer for time is daytime.\n2. Vehicle type extraction\nFrom the Summary, we can know the details of these two vehicles:\nV1: a 2002 Subaru Forester\nV2: a 2002 Buick Rendezvous\nV1 is a car and V2 is also a car.\n\nOutput:\n\n{'Weather': 'Sunny', 'Time': 'Daytime', 'Car type': 'Car'}"
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"What's your output for this summary:\n{summary}"
                    }
                ]
            }
        ],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        response_format={
            "type": "text"
        }
    )

    first_output = response.choices[0].message.content
    file_name = f"{record}_env_car_info.txt"
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(first_output)

def main():
    parser = argparse.ArgumentParser(description='MM ADS Testing - road type extraction')
    parser.add_argument('--data_path', default='E:\GitHub\CMIEA\Dataset', type=str, help='Path of MM crash dataset')
    args = parser.parse_args()

    # Get data ID list
    records = [name for name in os.listdir(args.data_path) if os.path.isdir(os.path.join(args.data_path, name))]

    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    result_folder = f"env_info_extract_experiment_{current_time}"
    os.makedirs(result_folder, exist_ok=True)
    folder_path = os.path.abspath(result_folder)

    for record in records:
        record_id = int(record)
        record_path = args.data_path + '\\' + record
        # Get encoded crash sketch
        # sketch = encode_image(record_path+ '\\' + 'Sketch.jpg')
        # Get crash summary
        with open(record_path+ '\\' + 'Summary.txt', 'r', encoding='utf-8') as file:
            summary = file.read()
        # Get response from GPT
        get_env(record, summary,folder_path)


    data = {}

    for filename in os.listdir(folder_path):
        if filename.endswith('_env_car_info.txt'):
            file_id = filename.split('_')[0]

            with open(os.path.join(folder_path, filename), 'r') as file:
                content = file.read()

                match = re.search(r'\{.*\}', content)
                if match:
                    info_dict = ast.literal_eval(match.group(0))
                    data[file_id] = info_dict
    print(data)
    data_path = folder_path + '\env_car_info.pkl'
    with open(data_path, 'wb') as pkl_file:
        pickle.dump(data, pkl_file)


if __name__=='__main__':
    main()