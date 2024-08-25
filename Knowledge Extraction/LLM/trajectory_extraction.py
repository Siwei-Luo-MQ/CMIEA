import pickle
from openai import OpenAI
import argparse
from datetime import datetime
import os
import pandas as pd
import base64
import re
import json
import pickle
import ast


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def get_traj(trajectory_training_data,example_sketch,sketch,summary,road_attribute,record,folder_path):
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are TrackMate, and you can predict vehicle trajectories based on the starting and ending points extracted from the crash report. "
                                "I will provide you with some vehicle trajectory data (composed of two-dimensional coordinate points). "
                                "You need to use this trajectory data as training data to learn how to predict vehicle trajectories."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "I have a set of multi-modal crash data, with each entry consisting of a crash sketch and a crash summary.\n"
                                "The crash sketch is a diagram which includes the road structure and vehicle movement information. "
                                "In the sketch, black solid lines generally represent the road edges, while black dashed lines typically indicate road dividers. "
                                "Vehicles in the sketch are usually depicted as rectangular blocks with labels such as V1 or V2, which are used to distinguish between different vehicles.\n"
                                "The crash summary is a textual description of the accident, including details about the road network, traffic participants, and environmental conditions.\n\n"
                                "In this task, I will provide you with the crash sketch, crash summary, and road network information (including details like road type, length, width, etc.). "
                                "There are 5 types of Road types: Straight, Intersection, T-intersection, Curve and Merge.\n\n"
                                "Your task is to extract the starting point and crash point of each traffic participant from the crash sketch, predict the vehicle's trajectory between the starting point and the crash point, and then verify the accuracy of your predicted trajectory based on the vehicle behavior described in the crash summary. \n\nFinally, you need to output the formatted vehicle trajectory information and verification results. \n\nBefore extracting the vehicle's starting point, collision point, and predicting its trajectory, you first need to define a two-dimensional Cartesian coordinate system. Then, map the road and vehicle information from the sketch onto this coordinate system before proceeding with the trajectory extraction.\n\nTo help you learn trajectory prediction, I will send you a batch of data for training:"
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": trajectory_training_data
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Here's an example of this task, you can use it to learn the whole process:\nInputs:\ncrash sketch:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{example_sketch}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "crash summary:\nVehicle one (V1 - case vehicle). a 2000 Chevrolet Malibu, 4-door sedan was traveling north in the right northbound lane of a limited access, four-lane, divided freeway (two lanes northbound, open median, two lanes southbound). Vehicle two (V2), a 1998 International tractor-trailer was traveling north in the left northbound lane of the same roadway but was slightly behind V1. For an unknown reason, the driver of V1 suddenly veered to the left and into the left northbound lane. The driver of V2 could not avoid V1 and the front of V2 struck the left-side of V1. The 63-year-old male driver of V1 (case occupant) was not using the available 3-point seat belt and no airbags deployed. This vehicle is equipped with frontal-impact airbags that did not deploy, but is not equipped with side-impact airbags. The driver of V1 was transported to a regional level-one trauma center via helicopter. Both vehicles were reportedly towed due to disabling damage. The driver of V1 expired 3 months post-crash due to a pre-existing illness not attributed to the crash.\nRoad Network: \n{'Road type': 'Straight', 'No_ways': 2, 'No_lanes': 4, 'Length': 50, 'Width': 4}\n\nAnalysis process:\n\nFrom the provided Road Network information, we know this is a straight road with four lanes, two in each direction, with a total length of 50 meters and a lane width of 4 meters each.\n\nWhen constructing a Cartesian coordinate system for the straight road, we need to define the bottom-left corner of the road as the origin. The X-axis is set horizontally with the positive direction to the right, and the Y-axis is set vertically with the positive direction upwards.\n\nBased on the sketch, we know there are two vehicles involved in the accident. We will use the center position of each vehicle to obtain the trajectory information.\n\nCombining the sketch and the summary information, we know both vehicles started from the left side of the sketch but were in different lanes. V1 was traveling in the lane closer to the X-axis compared to V2.\n\nWe can also infer that the two vehicles collided roughly in the middle of the road, with V1 moving into V2’s lane.\n\nUsing this information, we can deduce the starting positions and collision points for both vehicles. We create a list for each vehicle to store the starting point and collision point, with the starting point first and the collision point second.\n\nV1_traj = [(4,2), (25,6)]\nV2_traj = [(10,6), (25,6)]\n\nNext, you need to use the trajectory prediction model you learned to predict the trajectories of both vehicles between their starting points and collision points. Each vehicle's trajectory should include three intermediate points.\n\nIn this case, I will directly provide the answer for demonstration:\n\nV1_traj = [(4,2), (10,3), (15, 4), (20, 5), (25,6)]\nV2_traj = [(10,6), (13,6), (18,7), (21, 6), (25,6)]\n\nThen, you need to validate your predicted trajectories based on the descriptions of the vehicles’ trajectories in the summary to determine if your predictions are correct.\n\nOutput:\n{'V1_traj': [(4,2), (10,3), (15, 4), (20, 5), (25,6)], 'V2_traj': [(10,6), (13,6), (18,7), (21, 6), (25,6)], 'Validation': 'Pass' }"
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "To summarize and process the provided crash sketch and summary, we'll follow these steps:\n\n1. **Define the Cartesian Coordinate System**: \n   - The bottom-left corner of the road will be the origin (0, 0).\n   - The positive X-axis will be horizontal (right), and the positive Y-axis will be vertical (upwards).\n\n2. **Extract Starting Points and Collision Points**:\n   - Vehicle V1 travels north in the right lane, and based on the sketch, we will place its starting point towards the left side of the road.\n   - Vehicle V2 travels in the left lane but is slightly behind V1.\n   - The collision occurs when V1 moves into V2's lane.\n\n3. **Determine Starting and Collision Points**:\n   - For V1: Starting point at (4, 2) and collision point at an estimated point in the middle, (25, 6).\n   - For V2: Starting point at (10, 6) and collision point at the same point as V1, (25, 6).\n\n4. **Predict Trajectories**: Using estimation, we predict the intermediate points:\n   - **V1 trajectory**: Start at (4, 2), with intermediate points (10, 3), (15, 4), (20, 5), and end at (25, 6).\n   - **V2 trajectory**: Start at (10, 6), with intermediate points (13, 6), (18, 7), (21, 6), and end at (25, 6).\n\n5. **Validation**: Based on the summary, both vehicles’ predicted behaviors match the description of the collision dynamics.\n\n### Output Summary:\n```json\n{\n    'V1_traj': [(4, 2), (10, 3), (15, 4), (20, 5), (25, 6)],\n    'V2_traj': [(10, 6), (13, 6), (18, 7), (21, 6), (25, 6)],\n    'Validation': 'Pass'\n}\n```\n\nThis structured output encapsulates the trajectory details and confirms the trajectory predictions met the conditions described in the crash summary."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "When executing the task, please pay attention to the following points:\nIf the road type is Straight, you should first ensure that the straight road is aligned horizontally. If it is not, you need to rotate the image to make sure the road is horizontal. Then, map the bottom-left corner of the straight road in the sketch to (0,0), establish the X-axis along the horizontal direction (with right as the positive direction), and establish the Y-axis along the vertical direction (with upward as the positive direction).\nIf the road type is Intersection, you need to map the center of the intersection to (0,0), establish the X-axis along the horizontal direction (with right as the positive direction), and establish the Y-axis along the vertical direction (with upward as the positive direction).\nIf the road type is T-intersection, you should first ensure that the main road is horizontal and the side road is vertical, with the side road located below the main road. Then, map the center point where the main road and side road intersect to (0,0), establish the X-axis along the horizontal direction (with right as the positive direction), and establish the Y-axis along the vertical direction (with upward as the positive direction).\nIf the road type is Curve, use the bottom-left point of the leftmost end of the curve as (0,0).\nIf the road type is Merge, use the top-left point of the main road as (0,0)."
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Got it! I'll ensure to accurately follow the specified mapping guidelines for different road types. Let’s apply these rules to the trajectory extraction, prediction, and validation process.\n\n### Analysis Steps Based on Road Type\n\n1. **Determine Road Type**: From the provided input, we'll check the road type.\n\n2. **Mapping**:\n   - **Straight Road**: Ensure the road is horizontally aligned. Map the bottom-left corner to (0, 0), establish the X-axis rightward, and the Y-axis upward.\n   - **Intersection**: Map the center to (0, 0).\n   - **T-Intersection**: Ensure the main road is horizontal and the side road is vertical, then map the intersection point to (0, 0).\n   - **Curve**: Use the bottom-left point of the leftmost end of the curve as (0, 0).\n   - **Merge**: Use the top-left point of the main road as (0, 0).\n\n3. **Extract Start and Collision Points**: Based on the mapped coordinates and the sketch details, deduce the starting point and collision point for each vehicle.\n\n4. **Predict Trajectories**: Generate intermediate points representing the vehicles' paths between starting points and collision points.\n\n5. **Validate**: Compare the predicted trajectories against the behaviors described in the crash summary.\n\n### Example Process\nAssuming the provided example was of a straight road:\n\n1. **Road Type**: **Straight**\n2. **Mapping Determination**:\n   - Ensure road alignment.\n   - Bottom-left corner mapped to (0, 0).\n\n3. **Starting and Collision Points** (hypothetical):\n   - V1: Start at (4, 2), End at (25, 6).\n   - V2: Start at (10, 6), End at (25, 6).\n\n4. **Predicted Trajectories**:\n   - V1: [(4, 2), (10, 3), (15, 4), (20, 5), (25, 6)]\n   - V2: [(10, 6), (13, 6), (18, 7), (21, 6), (25, 6)]\n\n5. **Validation**: Check if both vehicles’ motions align with the summary description.\n\n### Output Summary\n```json\n{\n    'V1_traj': [(4, 2), (10, 3), (15, 4), (20, 5), (25, 6)],\n    'V2_traj': [(10, 6), (13, 6), (18, 7), (21, 6), (25, 6)],\n    'Validation': 'Pass'\n}\n```\n\nFor any new data or cases you provide, I will execute this process according to the specified guidelines."
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
                        "text": f'Summary: {summary}'
                    },
                    {
                        "type": "text",
                        "text": f'Road Network: {road_attribute}'
                    }
                ]
            }
        ],
        temperature=1,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        response_format={
            "type": "text"
        }
    )

    first_output = response.choices[0].message.content
    # save results
    file_name = f"{record}_trajectory.txt"
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(first_output)

def main():
    parser = argparse.ArgumentParser(description='MM ADS Testing - road network extraction')
    parser.add_argument('--data_path', default='E:\GitHub\CMIEA\Dataset', type=str, help='Path of MM crash dataset')
    parser.add_argument('--road_type_label', default='E:\GitHub\CMIEA\\road_type.xlsx', type=str,
                        help='Path of road type label file')
    parser.add_argument('--road_network', default='E:\GitHub\CMIEA\Knowledge Extraction\LLM\\road_network_extract_experiment_2024-08-25_17-25-00\\road_network_results.pkl', type=str,
                        help='Path of road type label file')
    parser.add_argument('--trajectory_training_data',
                        default="E:\GitHub\CMIEA\Knowledge Extraction\LLM\\trajectory_traing\\traffic_trajectory_0.pkl",
                        type=str,
                        help='Path of pracjectory training data folder')

    args = parser.parse_args()

    # Get original dataset
    records = [name for name in os.listdir(args.data_path) if os.path.isdir(os.path.join(args.data_path, name))]

    # Get road type labels
    df = pd.read_excel(args.road_type_label)
    road_type_labels = dict(zip(df['ID'], df['Label']))

    # Get road networks
    with open(args.road_network, 'rb') as file:
        road_network = pickle.load(file)

    # Get training data
    with open(args.trajectory_training_data, 'rb') as file:
        trajectory_training_data = pickle.load(file)
    trajectory_training_data = str(trajectory_training_data)

    # Create result folder
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    result_folder = f"trajectory_extract_experiment_{current_time}"
    os.makedirs(result_folder, exist_ok=True)
    folder_path = os.path.abspath(result_folder)

    for record in records:
        # data preparation
        record_id = int(record)
        record_path = args.data_path + '\\' + record

        # Get encoded crash sketch
        sketch = encode_image(record_path + '\\' + 'Sketch.jpg')

        # Get crash summary
        with open(record_path + '\\' + 'Summary.txt', 'r', encoding='utf-8') as file:
            summary = file.read()

        road_type = road_type_labels[record_id]

        road_attribute =road_network[record]
        road_attribute['Road type'] = road_type

        example_sketch = encode_image('E:\GitHub\CMIEA\Knowledge Extraction\LLM\sketch_example_1.jpg')
        # Get trajectory from GPT
        get_traj(trajectory_training_data,example_sketch,sketch,summary,road_attribute,record,folder_path)
        print(f'Record: {record} finished!')



if __name__=='__main__':
    main()