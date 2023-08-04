import pandas as pd
import json

raw_data_path = "../raw_data/MedicationQA.xlsx"
data = pd.read_excel(raw_data_path, index_col=None)
print(data.shape)

json_data = data.to_json(orient='records')
parsed_data = json.loads(json_data)

# Define the file path where you want to save the JSON data
file_path = '../data/MedicationQA.json'

# Write the JSON data to the file
with open(file_path, 'w') as json_file:
    json.dump(parsed_data, json_file)

print("JSON data saved to '{}'",file_path)
