import pandas as pd
import xml.etree.ElementTree as ET
import json

tree1 = ET.parse('../raw_data/LiveQA/TrainingDatasets/TREC-2017-LiveQA-Medical-Train-1.xml')
tree2 = ET.parse('../raw_data/LiveQA/TrainingDatasets/TREC-2017-LiveQA-Medical-Train-2.xml')
root1 = tree1.getroot()
root2 = tree2.getroot()

data = []

for nlm_question in root1.findall('NLM-QUESTION'):
    item_data = {}
    item_data['subject'] = nlm_question.find('SUBJECT').text
    item_data['Question'] = nlm_question.find('MESSAGE').text
    sub_question = nlm_question.find('SUB-QUESTIONS/SUB-QUESTION')
    item_data['focus'] = sub_question.find('ANNOTATIONS/FOCUS').text
    item_data['type'] = sub_question.find('ANNOTATIONS/TYPE').text
    item_data['Answer'] = sub_question.find('ANSWERS/ANSWER').text
    data.append(item_data)
    
for nlm_question in root2.findall('NLM-QUESTION'):
    item_data = {}
    item_data['subject'] = nlm_question.find('SUBJECT').text
    item_data['Question'] = nlm_question.find('MESSAGE').text
    sub_question = nlm_question.find('SUB-QUESTIONS/SUB-QUESTION')
    item_data['focus'] = sub_question.find('ANNOTATIONS/FOCUS').text
    item_data['type'] = sub_question.find('ANNOTATIONS/TYPE').text
    item_data['Answer'] = sub_question.find('ANSWERS/ANSWER').text
    data.append(item_data)

df = pd.DataFrame(data)

json_data = df.to_json(orient='records')
parsed_data = json.loads(json_data)

# Define the file path where you want to save the JSON data
file_path = '../data/LiveQA.json'

# Write the JSON data to the file
with open(file_path, 'w') as json_file:
    json.dump(parsed_data, json_file)

print("JSON data saved to '{}'",file_path)


