from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.vehicle.vehicle_type import DefaultVehicle
from metadrive.utils import generate_gif
from metadrive.policy.idm_policy import IDMPolicy, TrajectoryIDMPolicy
from metadrive.policy.expert_policy import ExpertPolicy
from metadrive.utils.draw_top_down_map import draw_top_down_map
import matplotlib.pyplot as plt
from metadrive.component.lane.straight_lane import StraightLane
from metadrive.scenario.parse_object_state import parse_object_state, get_idm_route, get_max_valid_indicis
import argparse
import os
import yaml

def run_straight():
    scenario_config = {'map_config': {'type': 'block_sequence',
                                      'config': 'S',
                                      'lane_width': 3,
                                      'lane_num': 1,
                                      'start_position': [0, 0],
                                      },
                       'traffic_density': 0,
                       'vehicle_config': {
                           'spawn_position_heading': [[0, 0], 0],
                       },
                       'use_render': True,
                       "truncate_as_terminate": True,
                       "crash_vehicle_done": True,
                       }
    pass

def run_intersection():
    pass

def run_T_intersection():
    pass

def run_curve():
    pass

def run_merge():
    pass

def load_yaml_files_to_dict(directory_path):
    yaml_dict = {}

    for filename in os.listdir(directory_path):
        if filename.endswith(".yaml"):
            file_id = os.path.splitext(filename)[0]
            file_path = os.path.join(directory_path, filename)

            with open(file_path, 'r', encoding='utf-8') as file:
                content = yaml.safe_load(file)
                yaml_dict[file_id] = content

    return yaml_dict

def main():
    parser = argparse.ArgumentParser(description='MM ADS Testing - road type extraction')
    parser.add_argument('--dsl_path', default=r'C:\Users\Kris\Desktop\CMIEA\Knowledge Extraction\LLM\Encoded_DSL_20240826_234220', type=str)
    args = parser.parse_args()

    directory_path = args.dsl_path  # 将这里替换为实际的路径
    yaml_data = load_yaml_files_to_dict(directory_path)

    print(yaml_data['100237'])

    # loop over all the cases
    for ID, DSL in yaml_data.items():
        actors = DSL['Actors']
        env = DSL['Env']
        road_network = DSL['Road network']
        road_type = DSL['Road type']
        if road_type == 'Straight':
            run_straight()
        elif road_type == 'Intersection':
            run_intersection()
        elif road_type == 'T-intersection':
            run_T_intersection()
        elif road_type == 'Curve':
            run_curve()
        elif road_type == 'Merge':
            run_merge()

if __name__=='__main__':
    main()