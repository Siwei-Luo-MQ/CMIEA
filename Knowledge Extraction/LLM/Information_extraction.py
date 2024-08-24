from openai import OpenAI
import argparse
from datetime import datetime
import os
import pandas as pd
import base64
import re


# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def road_type_extraction(sketch,summary,true_label,road_type_sketch,client):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "I need you to work as an assistant."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "I have a multi-modality dataset about car crashes, each crash contains a crash sketch and a crash summary. "
                                "Crash sketches are in graph modality while crash summaries are text data. The sketches visualise road networks, traffic participants, trajectories, and additional crash background information. "
                                "The crash summary is the text description of the crash, it contains road networks, traffic participants, trajectories, and environmental information.\n\n"
                                "Now, I need you to extract the road networks from the dataset. Your answers must be within these five values: Intersection, T-intersection, Straight, Curve, and Merge. Also, you need to return the validation result.\n\n"
                                "I will first show you an example to help you understand this task."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Sketch:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{road_type_sketch}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Summary: Vehicle one (V1 - case vehicle), a 2002 Subaru Forester, 4-door utility vehicle was traveling west in the westbound lane of a two-lane, two-way roadway and was negotiating a right curve. Vehicle two (V2), a 2002 Buick Rendezvous, 4-door utility vehicle was traveling east in the eastbound lane of the same roadway and was negotiating a left curve. It was daylight, snowing, and the bituminous road was icy. The driver of V1 lost control on the icy road and began to rotate counterclockwise. V1 rotated approximately 90 degrees, crossed the centerline and entered the eastbound lane. The driver of V2 could not avoid V1 and the front of V2 struck the right side of V1 in a T-type configuration. Both vehicles were towed due to disabling vehicle damage. The 36-year-old male driver of V1 (case occupant) was using the available three-point seat belt but no airbags deployed in the driver's seating position. He was transported via ground ambulance to a regional level-one trauma center.\n\nAnalysis process:\n\nStep 1 - Crash Summary Analysis\nBy analyzing the crash summary, we can know that:\n\tV1 was travelling on a curved, two-lane, two-way road, and \n\tV2 was also travelling on this curved road.\nTherefore, based on the road feature description provided in the crash summary, we can initially conclude that this is a curved road.\n\nStep 2 - Validation \n\nFrom the crash sketch, we can see two curved roads depicted, with no intersections on the far left or right sides of the roads.\nTherefore, based on the information in the sketch, we can verify the answer obtained in the first step. \nThe answer obtained in the first step has passed the verification.\n\nOutput:\n{'Road type': Curve, 'Validation': Pass}\n\nNote: If a case fails validation, you should output Failed in the validation result."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's your output for this case?\nSketch:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{sketch}"
                        }
                    },
                    {
                        "type": "text",
                        "text": summary
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
    print(first_output)
    road_type = re.search(r"'Road type': ([\w-]+)", first_output).group(1)
    validation = re.search(r"'Validation': (\w+)", first_output).group(1)
    pre_result = 1 if true_label == road_type else 0
    return road_type, validation, pre_result


def main():
    parser = argparse.ArgumentParser(description='MM ADS Testing - Information Extraction')
    parser.add_argument('--data_path',default='E:\GitHub\CMIEA\Dataset',type=str,help='Path of MM crash dataset')
    parser.add_argument('--road_type_label', default='E:\GitHub\CMIEA\\road_type.xlsx', type=str, help='Path of road type label file')
    args=parser.parse_args()

    # Get current time
    now=datetime.now()

    # Get data ID list
    records = [name for name in os.listdir(args.data_path) if os.path.isdir(os.path.join(args.data_path, name))]
    # Get road type labels
    df = pd.read_excel(args.road_type_label)
    road_type_labels = dict(zip(df['ID'], df['Label']))

    client = OpenAI()
    for record in records[3:]:
        record_id = int(record)
        record_path = args.data_path + '\\' + record
        # Get encoded crash sketch
        sketch = encode_image(record_path+ '\\' + 'Sketch.jpg')
        # Get crash summary
        with open(record_path+ '\\' + 'Summary.txt', 'r', encoding='utf-8') as file:
            summary = file.read()

        # Information Extraction Stage - Step 1
        # Extract road type from Summary and Validate using sketch
        # Get training sketch
        road_type_sketch = encode_image('E:\GitHub\CMIEA\Prompt Engineering\\road_type\Sketch.jpg')
        road_type, road_type_val, pre_result = road_type_extraction(sketch, summary, road_type_labels[record_id],road_type_sketch,client)
        print(road_type, road_type_val, pre_result)







if __name__=='__main__':
    main()
