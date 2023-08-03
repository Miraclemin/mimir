import json

# Step 1: Read the JSON file
file_path = '../raw_data/PubMedQA/ori_pqal.json'

with open(file_path, 'r') as json_file:
    data = json.load(json_file)

# Step 2 and 3: Extract inner dictionaries from the main dictionary
inner_dicts = [data[key] for key in data]

# Step 4: Convert inner dictionaries to a list of dictionaries
result_list = [inner_dict for inner_dict in inner_dicts]

# Step 5: Convert the list to a JSON-formatted string
json_string = json.dumps(result_list)

# Step 6: Write the JSON string to a file
output_file_path = '../data/PubMedQA.json'
with open(output_file_path, 'w') as json_file:
    json_file.write(json_string)

print("JSON file saved successfully.")
print(result_list)
