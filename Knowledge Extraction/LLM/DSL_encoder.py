import pandas as pd
import pickle
import os
import re
import yaml
from shutil import copyfile
from datetime import datetime

# 读取 XLSX 文件
path_A = r'C:\Users\Kris\Desktop\CMIEA\Knowledge Extraction\LLM\road_type_extract_experiment_2024-08-25_17-22-28\road_type_prediction.xlsx'
df_labels = pd.read_excel(path_A)
id_label_dict = dict(zip(df_labels['ID'], df_labels['Label']))

# 读取 road network 信息
path_B = r'C:\Users\Kris\Desktop\CMIEA\Knowledge Extraction\LLM\road_network_extract_experiment_2024-08-25_17-25-00\road_network_results.pkl'
with open(path_B, 'rb') as f:
    road_network_dict = pickle.load(f)

# 提取车辆轨迹和验证结果
def extract_trajectory_and_validation(file_content):
    traj_pattern = r'\'V\d+_traj\': \[(.*?)\]'
    validation_pattern = r'\'Validation\': \'(\w+)\''

    traj_matches = re.findall(traj_pattern, file_content)
    validation_match = re.search(validation_pattern, file_content)

    trajectory_data = {}
    for idx, traj in enumerate(traj_matches, start=1):
        traj_list = eval(f"[{traj}]")  # 将字符串转换为列表
        trajectory_data[f'V{idx}_traj'] = traj_list

    validation_result = validation_match.group(1) if validation_match else None
    trajectory_data['Validation'] = validation_result

    return trajectory_data

# 读取 TXT 文件夹中的所有文件
path_C = r'C:\Users\Kris\Desktop\CMIEA\Knowledge Extraction\LLM\trajectory_extract_experiment_2024-08-25_17-29-50'
trajectory_dict = {}

for txt_file in os.listdir(path_C):
    if txt_file.endswith('.txt'):
        file_id = txt_file.split('_')[0]
        with open(os.path.join(path_C, txt_file), 'r', encoding='utf-8') as f:
            file_content = f.read()
            trajectory_dict[file_id] = extract_trajectory_and_validation(file_content)

# 读取 Weather, Time, Car type 信息
path_D = r'C:\Users\Kris\Desktop\CMIEA\Knowledge Extraction\LLM\env_info_extract_experiment_2024-08-26_15-53-42\env_car_info.pkl'
with open(path_D, 'rb') as f:
    additional_info_dict = pickle.load(f)

# 获取当前时间，并创建包含时间戳的文件夹
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(os.getcwd(), f'Encoded_DSL_{current_time}')
os.makedirs(output_dir, exist_ok=True)

# 定义 YAML 模板文件路径
template_files = {
    'Merge': r'C:\Users\Kris\Desktop\CMIEA\Knowledge Extraction\LLM\DSL\Merge.yaml',
    'Straight': r'C:\Users\Kris\Desktop\CMIEA\Knowledge Extraction\LLM\DSL\Straight.yaml',
    'T-intersection': r'C:\Users\Kris\Desktop\CMIEA\Knowledge Extraction\LLM\DSL\T-intersection.yaml',
    'Curve': r'C:\Users\Kris\Desktop\CMIEA\Knowledge Extraction\LLM\DSL\Curve.yaml',
    'Intersection': r'C:\Users\Kris\Desktop\CMIEA\Knowledge Extraction\LLM\DSL\Intersection.yaml'
}

# 为每个 ID 生成 YAML 文件
for id_, label in id_label_dict.items():
    template_path = template_files.get(label)
    if template_path:
        # 复制模板文件到输出目录
        output_file = os.path.join(output_dir, f'{id_}.yaml')
        copyfile(template_path, output_file)

        # 读取复制后的 YAML 文件内容
        with open(output_file, 'r') as f:
            yaml_content = yaml.safe_load(f)

        # 填充 Environment 信息
        yaml_content['Environment'] = {
            'Weather': additional_info_dict.get(id_, {}).get('Weather', ''),
            'Time': additional_info_dict.get(id_, {}).get('Time', '')
        }

        # 填充 Road_Network 信息
        yaml_content['Road_Network'] = road_network_dict.get(id_, {})
        # 填充 Actors 信息
        actors = []
        for vehicle_id, trajectory in trajectory_dict.get(id_, {}).items():
            if 'traj' in vehicle_id:
                actor = {
                    'Type': additional_info_dict.get(id_, {}).get('Car type', ''),
                    'Trajectory': trajectory
                }
                actors.append(actor)
        yaml_content['Actors'] = actors

        # 将更新后的内容写回 YAML 文件
        with open(output_file, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
