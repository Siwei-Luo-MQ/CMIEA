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


def main():
    parser = argparse.ArgumentParser(description='MM ADS Testing - road network extraction')
    parser.add_argument('--data_path', default='E:\GitHub\CMIEA\Dataset', type=str, help='Path of MM crash dataset')
    parser.add_argument('--road_type_label', default='E:\GitHub\CMIEA\\road_type.xlsx', type=str,
                        help='Path of road type label file')
    parser.add_argument('--road_network', default='E:\GitHub\CMIEA\Knowledge Extraction\LLM\\road_network_extract_experiment_2024-08-25_13-57-52\\road_network_results.pkl', type=str,
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





if __name__=='__main__':
    main()