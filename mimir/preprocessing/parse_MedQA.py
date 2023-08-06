import json

# Step 1: Read the file
file_path = '../raw_data/MedQA/questions/US/train.jsonl'
with open(file_path, 'r') as file:
    data = file.read()

# Step 2: Split the data into individual JSON objects
json_objects = data.strip().split('\n')

# Step 3: Parse the JSON objects
parsed_data = [json.loads(obj) for obj in json_objects]

# Step 3.5: unify key name for all datasets
for data in parsed_data:
    q_value_to_move = data['question']
    a_value_to_move = data['answer']
    data['Question'] = q_value_to_move
    data['Answer'] = a_value_to_move
    del data['question']
    del data['answer']

# Step 4: Write to a new JSON file
output_file_path = '../data/MedQA_new.json'
with open(output_file_path, 'w') as output_file:
    json.dump(parsed_data, output_file, indent=2)
