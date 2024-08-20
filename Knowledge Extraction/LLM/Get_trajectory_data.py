from metadrive.engine.asset_loader import AssetLoader
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.envs.scenario_env import ScenarioEnv
import matplotlib.pyplot as plt
import math
import os

# turn on this to enable 3D render.
threeD_render=True
# Nuplan dataset
nuscenes_data =  AssetLoader.file_path(AssetLoader.asset_path, "nuscenes", unix_style=False)

env = ScenarioEnv(
    {
        "manual_control": False,
        "reactive_traffic": False,
        "use_render": threeD_render,
        "agent_policy": ReplayEgoCarPolicy,
        "data_directory": nuscenes_data,
        "num_scenarios": 1
        # num_scenarios is used to determine how many scenarios are loaded from the datasets.
    }
)

o, _ = env.reset(seed=0)
# The env.reset(seed=scenario-index) tells the simulator to remove all existing objects created in last episode,
# create a new scenario with index scenario-index and start a new episode.
# As we only have one scenario loaded to the simulator, the scenario-index can only be 0 in this example.

all_objects = env.engine.get_objects()
object_list = []
object_dic = {}

for k, v in all_objects.items():
  object_list.append(v)
remove_list = []

# filter
for item in object_list:
    try:
        item.last_position
    except Exception as e:
        remove_list.append(item)

for item in remove_list:
    object_list.remove(item)

for item in object_list:
    object_dic[item] = []

frames = []
for i in range(1, 100000):
    o, r, tm, tc, info = env.step([1.0, 0.])
    frames.append(env.render(mode="top_down",film_size=(1200, 1200)))
    if i % 2 == 0:
        for obj in object_list:
            obj_pos_x, obj_pos_y = obj.last_position
            object_dic[obj].append((obj_pos_x, obj_pos_y))
    if tm or tc:
        break
env.close()

for key, value in object_dic.items():
    print(f'{key}: \n {value}')

def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

def calculate_total_path_length(coordinates):
    return calculate_distance(coordinates[0], coordinates[-1])

def find_large_movement_actors(object_dic, threshold):
    vehicle_act_dict = {}

    for actor, coordinates in object_dic.items():
        total_length = calculate_total_path_length(coordinates)
        if total_length > threshold:
            vehicle_act_dict[actor] = coordinates

    return vehicle_act_dict

threshold = 20.0

vehicle_act_dict = find_large_movement_actors(object_dic, threshold)

# Visualization
plt.figure(figsize=(8, 6))
for key, coordinates in vehicle_act_dict.items():
    x, y = zip(*coordinates)
    plt.scatter(x, y, label=key)
plt.title('Coordinate Points Visualization')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.grid(True)
plt.show()