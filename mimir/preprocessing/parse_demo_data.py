import json
import pandas as pd

dataset_list = ["MedQA","MedMCQA","PubMedQA","MMLU Clinical Topics","MedicationQA","LiveQA"]

for ds_key in dataset_list:
    if ds_key == "MedicationQA" or ds_key == "MedMCQA" or ds_key == "MedQA" or ds_key == "PubMedQA" or ds_key == "LiveQA":
        dataset_path = "./mimir/data/"+ ds_key +".json"
    elif ds_key == "MMLU Clinical Topics":
        dataset_path = "./mimir/data/MMLU_clinical_topics.json"
    with open(dataset_path, 'r') as json_file:
        json_data = json.load(json_file)
        pd_data = pd.DataFrame(json_data).head(50)
    demo_data = pd_data.to_json(orient='records')
    parsed_data = json.loads(demo_data)
    output_file_path = './mimir/data/' + ds_key + '_demo.json'
    with open(output_file_path, 'w') as json_file:
        json.dump(parsed_data, json_file)
    
    