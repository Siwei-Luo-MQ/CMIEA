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
import time
import math

import metadrive

def convert_string_to_list_of_lists(string):
    # 去掉字符串中的方括号
    string = string.strip('[]')
    # 使用 eval 函数将字符串转换为列表元组
    list_of_tuples = eval(string)
    # 将元组转换为列表
    list_of_lists = [list(item) for item in list_of_tuples]
    return list_of_lists


def calculate_azimuth_angle(coordinates):
    # 提取最前面的两个点形成向量
    p1, p2 = coordinates[0], coordinates[1]

    # 计算向量的分量
    vector_x = p2[0] - p1[0]
    vector_y = p2[1] - p1[1]

    # 计算方位角（弧度制）
    azimuth_angle = math.atan2(vector_y, vector_x)

    # 将方位角调整为 [0, 2π] 的范围
    if azimuth_angle < 0:
        azimuth_angle += 2 * math.pi

    return azimuth_angle


def determine_heading(points):
    # 判断列表是否有足够的点
    if len(points) < 2:
        raise ValueError("至少需要两个点来判断车头朝向")

    # 计算第一个和第二个点的X坐标差值
    x_difference = points[1][0] - points[0][0]

    # 根据差值判断朝向
    if x_difference > 0:
        return 0  # 车头朝右
    else:
        return 3.14  # 车头朝左


def run_straight(road_network,env_info,actors,ID):
    # MetaDrive doesn't have a set of weather system
    # Get daytime
    day_time = env_info['Time']
    if day_time == 'Daytime':
        day_time = '11:00'
    elif day_time == 'Nighttime':
        day_time = '19:00'

    # Get the first car from Actors and set it as ego car
    # Get car list
    traj_list = []
    for key, value in actors.items():
        test = str(key).split('_')
        if len(test) == 2:
            traj_list.append(convert_string_to_list_of_lists(value))
    No_actors = len(traj_list)
    ego_headings = calculate_azimuth_angle(traj_list[0])
    ego_start_point = traj_list[0][0]

    scenario_config = {'map_config': {'type': 'block_sequence',
                                      'config': 'S',
                                      'lane_width': road_network['Width'],
                                      'lane_num': int(road_network['No_lanes']/2),
                                      'exit_length': road_network['Length'],
                                      'start_position': [0, 0]
                                      },
                       'traffic_density': 0,
                       'vehicle_config': {
                           'spawn_position_heading': [ego_start_point, ego_headings],
                       },
                       'use_render': True,
                       'daytime': day_time,
                       "truncate_as_terminate": True,
                       "crash_vehicle_done": True,
                       }

    env = MetaDriveEnv(scenario_config)
    frames = []

    ego_traj = get_idm_route(traj_list[0])
    npc_traj = get_idm_route(traj_list[1])

    try:
        env.reset()
        cfg = env.config["vehicle_config"]
        cfg["navigation"] = None  # it doesn't need navigation system

        npc = env.engine.spawn_object(DefaultVehicle,
                                      vehicle_config=cfg,
                                      position=traj_list[1][0],
                                      heading=calculate_azimuth_angle(traj_list[1]))

        env.engine.add_policy(npc.id, TrajectoryIDMPolicy, npc, env.engine.generate_seed(), npc_traj)
        env.engine.add_policy(env.agent.id, TrajectoryIDMPolicy, env.agent, env.engine.generate_seed(), ego_traj)

        for _ in range(100):
            p = env.engine.get_policy(npc.name)
            npc.before_step(p.act(True))
            _, r, _, _, info = env.step([0, 0])
            frame = env.render(mode="topdown",
                               window=False,
                               screen_size=(800, 400),
                               draw_target_vehicle_trajectory=False,
                               scaling=4,
                               camera_position=(10, 0))
            frames.append(frame)
            if info['crash']:
                break
        generate_gif(frames, gif_name=f"{ID}.gif")
    finally:
        env.close()

    time.sleep(2)

def run_intersection(road_network,env_info,actors,ID):
    # MetaDrive doesn't have a set of weather system
    # Get daytime
    day_time = env_info['Time']
    if day_time == 'Daytime':
        day_time = '11:00'
    elif day_time == 'Nighttime':
        day_time = '19:00'

    # Get center point
    scenario_config = {'map_config': {'type': 'block_sequence',
                                      'config': 'X',
                                      'lane_width': road_network['Width'],
                                      'lane_num': int(road_network['No_lanes'] / 2),
                                      'exit_length': road_network['Length'] / 2,
                                      },
                       'traffic_density': 0,
                       # 'vehicle_config': {
                       #     'spawn_position_heading': [ego_start_point, ego_headings],
                       # },
                       'use_render': True,
                       'daytime': day_time,
                       "truncate_as_terminate": True,
                       "crash_vehicle_done": True,
                       }
    env = MetaDriveEnv(scenario_config)
    env.reset()
    Road_Info = dict()
    for k, v in env.engine.current_map.road_network.graph.items():
        for k1, v1 in v.items():
            for lane in v1:
                Road_Info[k] = (float(lane.start[0]), float(lane.start[1]))
                Road_Info[k1] = (float(lane.end[0]), float(lane.end[1]))
    env.close()
    for label, (x, y) in Road_Info.items():
        if label == '->>>':
            left_up = [x, y]
        elif label == '1X1_0_':
            right_botm = [x, y]
    center = [((left_up[0] + right_botm[0]) / 2), ((left_up[1] + right_botm[1]) / 2)]

    # center shift
    # Get the first car from Actors and set it as ego car
    # Get car list
    traj_list = []
    for key, value in actors.items():
        test = str(key).split('_')
        if len(test) == 2:
            traj_list.append(convert_string_to_list_of_lists(value))
    for i in range(len(traj_list[0])):
        traj_list[0][i][0] = traj_list[0][i][0] + center[0]
        traj_list[0][i][1] = traj_list[0][i][1] + center[1]

    for i in range(len(traj_list[1])):
        traj_list[1][i][0] = traj_list[1][i][0] + center[0]
        traj_list[1][i][1] = traj_list[1][i][1] + center[1]

    ego_headings = calculate_azimuth_angle(traj_list[0])
    ego_start_point = traj_list[0][0]
    # start_position = [int(road_network['Length']/2),int((road_network['No_lanes'] / 2)*road_network['Width'])]

    scenario_config = {'map_config': {'type': 'block_sequence',
                                      'config': 'X',
                                      'lane_width': road_network['Width'],
                                      'lane_num': int(road_network['No_lanes'] / 2),
                                      'exit_length': road_network['Length']/2,
                                      },
                       'traffic_density': 0,
                       'vehicle_config': {
                           'spawn_position_heading': [ego_start_point, ego_headings],
                       },
                       'use_render': True,
                       'daytime': day_time,
                       "truncate_as_terminate": True,
                       "crash_vehicle_done": True,
                       }

    env = MetaDriveEnv(scenario_config)
    frames = []

    ego_traj = get_idm_route(traj_list[0])
    npc_traj = get_idm_route(traj_list[1])

    try:
        env.reset()
        cfg = env.config["vehicle_config"]
        cfg["navigation"] = None  # it doesn't need navigation system

        npc = env.engine.spawn_object(DefaultVehicle,
                                      vehicle_config=cfg,
                                      position=traj_list[1][0],
                                      heading=calculate_azimuth_angle(traj_list[1]))

        env.engine.add_policy(npc.id, TrajectoryIDMPolicy, npc, env.engine.generate_seed(), npc_traj)
        env.engine.add_policy(env.agent.id, TrajectoryIDMPolicy, env.agent, env.engine.generate_seed(), ego_traj)

        for _ in range(100):
            p = env.engine.get_policy(npc.name)
            npc.before_step(p.act(True))
            _, r, _, _, info = env.step([0, 0])
            frame = env.render(mode="topdown",
                               window=False,
                               screen_size=(800, 400),
                               draw_target_vehicle_trajectory=False,
                               scaling=4,
                               camera_position=(10, 0))
            frames.append(frame)
            if info['crash']:
                break
        generate_gif(frames, gif_name=f"{ID}.gif")
    finally:
        env.close()

    time.sleep(2)

def run_T_intersection(road_network,env_info,actors,ID):
    # MetaDrive doesn't have a set of weather system
    # Get daytime
    day_time = env_info['Time']
    if day_time == 'Daytime':
        day_time = '11:00'
    elif day_time == 'Nighttime':
        day_time = '19:00'


    # Get center point
    scenario_config = {'map_config': {'type': 'block_sequence',
                                      'config': 'T',
                                      'lane_width': road_network['Width'],
                                      'lane_num': int(road_network['No_lanes_main_road'] / 2),
                                      'exit_length': road_network['Length_main'] / 2,
                                      },
                       'traffic_density': 0,
                       # 'vehicle_config': {
                       #     'spawn_position_heading': [ego_start_point, ego_headings],
                       # },
                       'use_render': True,
                       'daytime': day_time,
                       "truncate_as_terminate": True,
                       "crash_vehicle_done": True,
                       }
    env = MetaDriveEnv(scenario_config)
    env.reset()
    Road_Info = dict()
    for k, v in env.engine.current_map.road_network.graph.items():
        for k1, v1 in v.items():
            for lane in v1:
                Road_Info[k] = (float(lane.start[0]), float(lane.start[1]))
                Road_Info[k1] = (float(lane.end[0]), float(lane.end[1]))
    env.close()
    for label, (x, y) in Road_Info.items():
        if label == '->>>':
            left_up = [x, y]
        elif label == '1T1_0_':
            right_botm = [x, y]
    center = [((left_up[0] + right_botm[0]) / 2), ((left_up[1] + right_botm[1]) / 2)]

    # center shift
    # Get the first car from Actors and set it as ego car
    # Get car list
    traj_list = []
    for key, value in actors.items():
        test = str(key).split('_')
        if len(test) == 2:
            traj_list.append(convert_string_to_list_of_lists(value))
    for i in range(len(traj_list[0])):
        traj_list[0][i][0] = traj_list[0][i][0] + center[0]
        traj_list[0][i][1] = traj_list[0][i][1] + center[1]

    for i in range(len(traj_list[1])):
        traj_list[1][i][0] = traj_list[1][i][0] + center[0]
        traj_list[1][i][1] = traj_list[1][i][1] + center[1]

    ego_headings = calculate_azimuth_angle(traj_list[0])
    ego_start_point = traj_list[0][0]
    # start_position = [int(road_network['Length']/2),int((road_network['No_lanes'] / 2)*road_network['Width'])]

    scenario_config = {'map_config': {'type': 'block_sequence',
                                      'config': 'T',
                                      'lane_width': road_network['Width'],
                                      'lane_num': int(road_network['No_lanes_main_road'] / 2),
                                      'exit_length': road_network['Length_main'] / 2,
                                      },
                       'traffic_density': 0,
                       'vehicle_config': {
                           'spawn_position_heading': [ego_start_point, ego_headings],
                       },
                       'use_render': True,
                       'daytime': day_time,
                       "truncate_as_terminate": True,
                       "crash_vehicle_done": True,
                       }

    env = MetaDriveEnv(scenario_config)
    frames = []

    ego_traj = get_idm_route(traj_list[0])
    npc_traj = get_idm_route(traj_list[1])

    print('-----------------------')
    print(scenario_config)
    print('-----------------------')
    print(traj_list[0])
    print('++++++++++')
    print(traj_list[1])
    print('++++++++++')

    try:
        env.reset()
        cfg = env.config["vehicle_config"]
        cfg["navigation"] = None  # it doesn't need navigation system

        npc = env.engine.spawn_object(DefaultVehicle,
                                      vehicle_config=cfg,
                                      position=traj_list[1][0],
                                      heading=calculate_azimuth_angle(traj_list[1]))

        env.engine.add_policy(npc.id, TrajectoryIDMPolicy, npc, env.engine.generate_seed(), npc_traj)
        env.engine.add_policy(env.agent.id, TrajectoryIDMPolicy, env.agent, env.engine.generate_seed(), ego_traj)

        for _ in range(100):
            p = env.engine.get_policy(npc.name)
            npc.before_step(p.act(True))
            _, r, _, _, info = env.step([0, 0])
            frame = env.render(mode="topdown",
                               window=False,
                               screen_size=(800, 400),
                               draw_target_vehicle_trajectory=False,
                               scaling=4,
                               camera_position=(10, 0))
            frames.append(frame)
            if info['crash']:
                break
        generate_gif(frames, gif_name=f"{ID}.gif")
    finally:
        env.close()

    time.sleep(2)

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

    directory_path = args.dsl_path
    yaml_data = load_yaml_files_to_dict(directory_path)

    # loop over all the cases
    for ID, DSL in yaml_data.items():
        actors = DSL['Actors']
        env_info = DSL['Env']
        road_network = DSL['Road network']
        road_type = DSL['Road type']
        if road_type == 'Straight':
            pass
            # run_straight(road_network,env_info,actors,ID)
        elif road_type == 'Intersection':
            # run_intersection(road_network,env_info,actors,ID)
            pass
        elif road_type == 'T-intersection':
            # if ID == '105203':
            #     run_T_intersection(road_network,env_info,actors,ID)
            pass
        elif road_type == 'Curve':
            run_curve()
        elif road_type == 'Merge':
            run_merge()

if __name__=='__main__':
    main()