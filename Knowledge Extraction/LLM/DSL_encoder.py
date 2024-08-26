import pandas as pd
import pickle
import os
import re
import ast
import yaml
from shutil import copyfile
from datetime import datetime

# 读取 XLSX 文件
path_A = r'.\road_type_extract_experiment_2024-08-25_17-22-28\road_type_prediction.xlsx'
df_labels = pd.read_excel(path_A)
id_label_dict = dict(zip(df_labels['ID'], df_labels['Label']))
print(id_label_dict)

# 读取 road network 信息
path_B = r'.\road_network_extract_experiment_2024-08-25_17-25-00\road_network_results.pkl'
with open(path_B, 'rb') as f:
    road_network_dict = pickle.load(f)
print(road_network_dict)

# 提取车辆轨迹和验证结果
def extract_trajectory_and_validation(file_content):
    # 正则表达式匹配车辆轨迹（支持单引号和双引号）
    traj_pattern = r'["\']V\d+_traj["\']: \[(.*?)\]'
    # 正则表达式匹配验证结果（支持单引号和双引号）
    validation_pattern = r'["\']Validation["\']: ["\'](\w+)["\']'

    traj_matches = re.findall(traj_pattern, file_content)
    validation_match = re.search(validation_pattern, file_content)

    trajectory_data = {}
    for idx, traj in enumerate(traj_matches, start=1):
        # 使用ast.literal_eval来安全地解析字符串
        traj_list = ast.literal_eval(f"[{traj}]")
        # trajectory_data[f'V{idx}_traj'] = traj_list
        trajectory_data[f'V{idx}_traj'] = str(traj_list)

    validation_result = validation_match.group(1) if validation_match else None
    trajectory_data['Validation'] = validation_result

    return trajectory_data


# 读取 TXT 文件夹中的所有文件
path_C = r'.\trajectory_extract_experiment_2024-08-25_17-29-50'
trajectory_dict = {}

for txt_file in os.listdir(path_C):
    if txt_file.endswith('.txt'):
        file_id = txt_file.split('_')[0]
        with open(os.path.join(path_C, txt_file), 'r', encoding='utf-8') as f:
            file_content = f.read()
            trajectory_dict[file_id] = extract_trajectory_and_validation(file_content)
print(trajectory_dict)

# 读取 Weather, Time, Car type 信息
path_D = r'.\env_info_extract_experiment_2024-08-26_15-53-42\env_car_info.pkl'
with open(path_D, 'rb') as f:
    additional_info_dict = pickle.load(f)
print(additional_info_dict)

# 获取当前时间，并创建包含时间戳的文件夹
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(os.getcwd(), f'Encoded_DSL_{current_time}')
os.makedirs(output_dir, exist_ok=True)

# 定义 YAML 模板文件路径
template_files = {
    'Merge': r'.\DSL\Merge.yaml',
    'Straight': r'.\DSL\Straight.yaml',
    'T-intersection': r'.\DSL\T-intersection.yaml',
    'Curve': r'.\DSL\Curve.yaml',
    'Intersection': r'.\DSL\Intersection.yaml'
}

def check_dict_lengths_equal(*dicts):
    lengths = [len(d) for d in dicts]
    return len(set(lengths)) == 1

are_lengths_equal = check_dict_lengths_equal(id_label_dict, road_network_dict, trajectory_dict, additional_info_dict)


if are_lengths_equal:
    for id_, label in id_label_dict.items():
        id_ = str(id_)
        encoded_data= {
            'Scenario': id_,
            'Road type': label,
            'Road network': road_network_dict[id_],
            'Actors': trajectory_dict[id_],
            'Env': additional_info_dict[id_]
        }
        file_name = output_dir + f'\{id_}.yaml'
        with open(file_name, 'w') as yaml_file:
            yaml.dump(encoded_data, yaml_file, default_flow_style=False, allow_unicode=True)

# 为每个 ID 生成 YAML 文件
# for id_, label in id_label_dict.items():
#     template_path = template_files.get(label)
#     if template_path:
#         # 复制模板文件到输出目录
#         output_file = os.path.join(output_dir, f'{id_}.yaml')
#         copyfile(template_path, output_file)
#
#         # 读取复制后的 YAML 文件内容
#         with open(output_file, 'r') as f:
#             yaml_content = yaml.safe_load(f)
#         id_ = str(id_)
#         # 填充 Environment 信息
#         yaml_content['Environment'] = {
#             'Weather': additional_info_dict.get(id_, {}).get('Weather', ''),
#             'Time': additional_info_dict.get(id_, {}).get('Time', '')
#         }
#
#         # 填充 Road_Network 信息
#         yaml_content['Road_Network'] = road_network_dict.get(id_, {})
#         # 填充 Actors 信息
#         actors = []
#         for vehicle_id, trajectory in trajectory_dict.get(id_, {}).items():
#             if 'traj' in vehicle_id:
#                 actor = {
#                     'Type': additional_info_dict.get(id_, {}).get('Car type', ''),
#                     'Trajectory': trajectory
#                 }
#                 actors.append(actor)
#         yaml_content['Actors'] = actors
#
#         # 将更新后的内容写回 YAML 文件
#         with open(output_file, 'w') as f:
#             yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
