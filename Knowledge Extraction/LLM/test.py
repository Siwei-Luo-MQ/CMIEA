import pickle

with open(r'C:\Users\Kris\Desktop\CMIEA\Knowledge Extraction\LLM\road_network_extract_experiment_2024-08-25_17-25-00\road_network_results.pkl','rb') as file:
    data = pickle.load(file)

print(data)