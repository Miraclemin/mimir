import pandas as pd
import json

# List all the dataset for MMLU Clinical Topics

anatomy_path_dev = "../raw_data/MMLU/dev/anatomy_dev.csv"
college_medicine_path_dev = "../raw_data/MMLU/dev/college_medicine_dev.csv"
clinical_knowledge_path_dev = "../raw_data/MMLU/dev/clinical_knowledge_dev.csv"
college_biology_path_dev = "../raw_data/MMLU/dev/college_biology_dev.csv"
medical_genetics_path_dev =  "../raw_data/MMLU/dev/medical_genetics_dev.csv"
professional_medicine_path_dev = "../raw_data/MMLU/dev/professional_medicine_dev.csv"
anatomy_path_test = "../raw_data/MMLU/test/anatomy_test.csv"
college_medicine_path_test = "../raw_data/MMLU/test/college_medicine_test.csv"
clinical_knowledge_path_test = "../raw_data/MMLU/test/clinical_knowledge_test.csv"
college_biology_path_test = "../raw_data/MMLU/test/college_biology_test.csv"
medical_genetics_path_test = "../raw_data/MMLU/test/medical_genetics_test.csv"
professional_medicine_path_test = "../raw_data/MMLU/test/professional_medicine_test.csv"

csv_files = [anatomy_path_dev,college_medicine_path_dev,clinical_knowledge_path_dev,college_biology_path_dev,medical_genetics_path_dev,professional_medicine_path_dev,professional_medicine_path_test,anatomy_path_test,college_medicine_path_test,clinical_knowledge_path_test,college_biology_path_test,medical_genetics_path_test]
dfs = []
# Loop through the files and read each CSV into a dataframe
for file_path in csv_files:
    df = pd.read_csv(file_path,header=None)
    dfs.append(df)

# Concatenate the dataframes by row
concatenated_df = pd.concat(dfs, ignore_index=True)



# concatenated_df['Answer'] = concatenated_df.iloc[:, 5].map({
#     "A": concatenated_df.iloc[:, 1],
#     "B": concatenated_df.iloc[:, 2],
#     "C": concatenated_df.iloc[:, 3],
#     "D": concatenated_df.iloc[:, 4]
# })

concatenated_df['Answer'] = concatenated_df.apply(lambda row: row[1] if row[5] == 'A'
                                else row[2] if row[5] == 'B'
                                else row[3] if row[5] == 'C'
                                else row[4] if row[5] == 'D'
                                else None, axis=1)

concatenated_df.rename(columns={concatenated_df.columns[0]: 'Question'}, inplace=True)
# Display the final concatenated dataframe
print(concatenated_df)

json_data = concatenated_df.to_json(orient='records')
parsed_data = json.loads(json_data)

# Define the file path where you want to save the JSON data
file_path = '../data/MMLU_clinical_topics.json'

# Write the JSON data to the file
with open(file_path, 'w') as json_file:
    json.dump(parsed_data, json_file)

print("JSON data saved to '{}'",file_path)

