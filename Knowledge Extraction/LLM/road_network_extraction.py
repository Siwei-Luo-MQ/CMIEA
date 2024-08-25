from openai import OpenAI
import argparse
from datetime import datetime
import os
import pandas as pd
import base64
import re
import json
import pickle

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def straight_road_network(example_sketch,sketch,summary,record,folder_path,road_type):
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "I need you to work as an assistant.\n\n"
                                "I have a sketch and a summary of a car crash. Crash sketches are in graph modality while crash summaries are text data. \n"
                                "In the sketches, road networks, traffic participants, trajectories, and additional crash background information are visualized. \n"
                                "The crash summary is the text description of the crash, it contains road networks, traffic participants, trajectories, and environmental information. \n\n"
                                "It is known that this car crash happened on a straight road.\n\n"
                                "You are required to extract the configuration details of this road network from the sketch and summary. To be more specific, you need to answer the following questions:\n"
                                "1. How many ways are there in the road? (your answer here should be an integer)\n"
                                "2. How many lanes in total in the road? (your answer here should be an integer)\n"
                                "3. What's the length of the road? (The unit is meter, you don’t need to include ‘meter’ in your answer)\n"
                                "4. What's the width of a single lane? (The unit is meter, you don’t need to include ‘meter’ in your answer)\n\n"
                                "To assist your calculation, here's some common knowledge you can use:\n"
                                "length of a car: 5 meters\n"
                                "width of a car: 2 meters\n"
                                "length of a truck: 20 meters\n"
                                "width of a truck: 2.5 meters\n\n"
                                "Your Output must in this structure: {'No_ways': <your answer for the number of ways>, 'No_lanes': <your answer for the number of lanes>, 'Length': <your answer for the length of the road>, 'Width': <your answer for the width of a single lane>}\n\n"
                                "Note: Your answers do not need to be enclosed in quotation marks\n\n"
                                "I will show you an example to help you better understand this task."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Crash sketch:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{example_sketch}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "\nSummary:\nVehicle one (V1 - case vehicle), a 1999 Ford Taurus, 4-door sedan was traveling east in the eastbound lane of a two-lane, two-way roadway. Vehicle two (V2), a 2002 Chevrolet Tahoe, 4-door utility vehicle was stopped at an angle, blocking the eastbound traffic lane. V2 is a marked police vehicle that had its overhead lights operating and the driver was out of the vehicle assisting other officers clear a previous crash. It was dark, the sky was clear, and the bituminous road was dry. The driver of V1 failed to completely clear the frost from her windshield and had a limited vision of the road. The driver of V1 did not see V2 and the front of V1 struck the right side of V2 in a T-type configuration. Both vehicles were towed due to disabling vehicle damage. The 66-year-old female driver of V1 (case occupant) was not using the available three-point seat belt and the steering-wheel airbag deployed. She was transported via ground ambulance to a local hosptial and later transferred to a regional level-one trauma center.\n\nAnalysis process:\n\nStep 1 - Extract the number of ways and lanes from the crash summary\nFrom the summary, we can know that the following sentences described the number of ways and lanes:\n\n'Vehicle one (V1 - case vehicle), a 1999 Ford Taurus, 4-door sedan was traveling east in the eastbound lane of a two-lane, two-way roadway.'\nFrom this sentence, we can know that the word - 'two-way' is used for describing the number of ways and the word - 'two-lane' is used for describing the number of lanes.\nTherefor, the answer for the first question (number of ways) is 2 and the answer for the second question (number of lanes) is 2.\n\nStep 2 - Approximate the length of the road and the width of the lane from the sketch\nFrom the sketch we can know the two ends of the straight road are located on both sides of the sketch, so we need to calculate the distance from the leftmost end to the rightmost end in the sketch.\nThe red rectangular block marked with V1 in the image is a car. From the sketch, we can estimate that the length of the straight road is approximately equal to 11 car body lengths.\nTherefor we can calculate the length of the road in this way:\n5 x 11 = 55 meters\n\nFrom the sketch, we can see that the red car is driving in the middle of a lane, and both sides of the car are about half the width of the car's body away from the edge of the lane.\nTherefor we can calculate the width of the lane in this way:\n2 + 2/2 + 2/2 = 4 meters\n\nFinally,  format the output:\n\n{'No_ways': 2, 'No_lanes': 2, 'Length': 55, 'Width': 4}"
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's your answer for this case?\nSketch:"
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
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        response_format={
            "type": "text"
        }
    )

    # Save original response
    first_output = response.choices[0].message.content
    network = json.loads(first_output)
    network['Road type'] = road_type
    file_name = f"{record}_road_network.pkl"
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, "wb") as file:
        pickle.dump(network, file)

def curve_road_network(example_sketch,sketch,summary,record,folder_path,road_type,curvature):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "I need you to work as an assistant.\n\nI have a sketch and a summary of a car crash. Crash sketches are in graph modality while crash summaries are text data. \nIn the sketches, road networks, traffic participants, trajectories, and additional crash background information are visualized. \nThe crash summary is the text description of the crash, it contains road networks, traffic participants, trajectories, and environmental information. \nIt is known that this car crash happened on a curve road.\n\nYou are required to extract the configuration details of this road network from the sketch and summary. To be more specific, you need to answer the following questions:\n1. How many ways are there in the road? (your answer here should be an integer)\n2. How many lanes are in total on the road? (your answer here should be an integer)\n3. What's the length of the road? (The unit is meter, you don’t need to include ‘meter’ in your answer)\n4. What's the width of a single lane? (The unit is meter, you don’t need to include ‘meter’ in your answer)\n5. What's the curvature of the road? (your answer should be 'small' or 'big')\n\nTo assist your calculation, here's some common knowledge you can use:\nlength of a car: 5 meters\nwidth of a car: 2 meters\nlength of a truck: 20 meters\nwidth of a truck: 2.5 meters\n\nYour Output must in this structure: {'No_ways': <your answer for the number of ways>, 'No_lanes': <your answer for the number of lanes>, 'Length': <your answer for the length of the road>, 'Width': <your answer for the width of a single lane>, 'Curvature': '<your answer for the curvature>'}\n\nNote: Except for the curvature, your answers do not need to be enclosed in quotation marks.\n\nI will first show you a picture to explain the size of curvature and then show you an example to help you better understand this task."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{curvature}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "From the picture you can see two curves, one with a small curvature, which we named C1, and the other with a large curvature, which we named C2. If you think the curvature of the road in the sketch is more similar to C1, then the answer to Question 5 is small, but if you think the curvature of the road in the sketch is more similar to C2, then the answer to Question 5 is big."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Crash sketch: "
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{example_sketch}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Crash summary: \nVehicle one (V1 - case vehicle), a 1998 Mercury Mountaineer, 4-door utility vehicle was traveling east in the eastbound lane of a two-lane, two-way rural roadway and was negotiating a slight curve to the left. Vehicle two (V2), a 2002 Ford F150, 4-door extended cab pickup truck was traveling west in the westbound lane of the same road but was negoatiating a slight curve to the right. It was daylight, the sky was cloudy, and the bituminous road was dry. The driver of V2, for an unknown reason, drifted to the left, crossed the centerline, and entered the eastbound lane of travel. The driver of V1 attempted to avoid V2 by steering right but the front of V2 struck the front of V1 in a narrow, offset-frontal configuration. After the impact, V1 veered off the right side of the road and came to rest facing south. V2 was redirected back into the westbound lane, but then traveled back across the eastbound lane, departed the left side of the road, and came to rest off the south side of the road facing southwest. Both vehicles were towed due to disabling vehicle damage. The 34-year-old female driver of V1 (case occupant) was using the available three-point seat belt and the steering-wheel airbag deployed. The 2-year-old right-rear male passenger was seated in a forward-facing child seat that was secured to the vehicle using the available three-point seat belt. Both occupants of V1 were transported via helicopter to a regional level-one trauma center.\nAnalysis process:\n\nStep 1 - Extract the number of ways and lanes from crash summary\nFrom the summary, we can know that the following sentences described the number of ways and lanes:\n\n'Vehicle one (V1 - case vehicle), a 1998 Mercury Mountaineer, 4-door utility vehicle was traveling east in the eastbound lane of a two-lane, two-way rural roadway and was negotiating a slight curve to the left. Vehicle two (V2), a 2002 Ford F150, 4-door extended cab pickup truck was traveling west in the westbound lane of the same road but was negoatiating a slight curve to the right.'\n\nFrom this sentence, we can know that these cars drive on the same road and the word - 'two-way' is used for describing the number of ways and the word - 'two-lane' is used for describing the number of lanes.\nTherefor, the answer for the first question (number of ways) is 2 and the answer for the second question (number of lanes) is 2.\n\nStep 2 - Approximate the length of the road and the width of the lane from the sketch\nWe define the straight-line distance between the two ends of a curve as the road length. From the sketch we can know the two ends of the curve road are located on both sides of the sketch, so we need to calculate the distance from the leftmost end to the rightmost end in the sketch.\nThe red rectangular block marked with V1 in the image is a car. From the sketch we can estimate that the straight-line distance between the two ends of this curve is approximately equal to 10 car body lengths.\nTherefor we can calculate the length of the road in this way:\nlength_road = 5 x 10 = 50 meters\n\nFrom the sketch, we can see that the red car is driving in the middle of a lane, and both sides of the car are about half the width of the car's body away from the edge of the lane.\nTherefor we can calculate the width of the lane in this way:\nwidth_road = 2 + 2/2 + 2/2 = 4 meters\n\nStep 3 - Approximate the curvature\nComparing the curvature of the curved road in the sketch with C1 and C2, the curvature of the road in the sketch is obviously smaller than that of C1, so curvature = small\n\nFinally, Format the output:\n{'No_ways': 2, 'No_lanes': 2, 'Length': 50, 'Width': 4, 'Curvature': 'small'}"
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's your answer for this case?\nSketch:"
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
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        response_format={
            "type": "text"
        }
    )

    # Save original response
    first_output = response.choices[0].message.content
    network = json.loads(first_output)
    network['Road type'] = road_type
    file_name = f"{record}_road_network.pkl"
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, "wb") as file:
        pickle.dump(network, file)

def intersection_road_network():
    pass

def t_intersection_road_network():
    pass

def merge_road_network():
    pass

def main():
    parser = argparse.ArgumentParser(description='MM ADS Testing - road network extraction')
    parser.add_argument('--data_path', default='E:\GitHub\CMIEA\Dataset', type=str, help='Path of MM crash dataset')
    parser.add_argument('--road_type_label', default='E:\GitHub\CMIEA\\road_type.xlsx', type=str,
                        help='Path of road type label file')
    args = parser.parse_args()

    # Get data ID list
    records = [name for name in os.listdir(args.data_path) if os.path.isdir(os.path.join(args.data_path, name))]
    # Get road type labels
    df = pd.read_excel(args.road_type_label)
    road_type_labels = dict(zip(df['ID'], df['Label']))

    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    result_folder = f"road_network_extract_experiment_{current_time}"
    os.makedirs(result_folder, exist_ok=True)
    folder_path = os.path.abspath(result_folder)

    for record in records:
        record_id = int(record)
        record_path = args.data_path + '\\' + record
        # Get encoded crash sketch
        sketch = encode_image(record_path + '\\' + 'Sketch.jpg')
        # Get crash summary
        with open(record_path + '\\' + 'Summary.txt', 'r', encoding='utf-8') as file:
            summary = file.read()

        # Information Extraction Stage - Step 2
        # Extract road network from Summary and sketch
        if road_type_labels[record_id] == 'Straight':
            example_sketch = encode_image(
                'E:\GitHub\CMIEA\Knowledge Extraction\Road Attribute Extraction\Straight\Sketch.jpg')
            straight_road_network(example_sketch,
                                  sketch,
                                  summary,
                                  record,
                                  folder_path,
                                  road_type_labels[record_id])
        elif road_type_labels[record_id] == 'Curve':
            example_sketch = encode_image(
                'E:\GitHub\CMIEA\Knowledge Extraction\Road Attribute Extraction\Curve\Sketch.jpg')
            curvature = encode_image(
                'E:\GitHub\CMIEA\Knowledge Extraction\Road Attribute Extraction\Curve\curvature.jpg')
            curve_road_network(example_sketch,
                                  sketch,
                                  summary,
                                  record,
                                  folder_path,
                                  road_type_labels[record_id],
                                  curvature
                               )
        elif road_type_labels[record_id] == 'Intersection':
            intersection_road_network()
        elif road_type_labels[record_id] == 'T-intersection':
            t_intersection_road_network()
        elif road_type_labels[record_id] == 'Merge':
            merge_road_network()




if __name__=='__main__':
    main()