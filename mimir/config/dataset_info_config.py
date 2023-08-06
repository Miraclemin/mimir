import json
from datasets import list_datasets
import streamlit as st
import webbrowser
# Create a dictionary with the dataset information
datasets_info = {
    "MedQA": {
        "description": "MedQA is a large-scale, multiple-choice question answering (MCQA) dataset designed for medical student exam preparation. It contains over 10,000 questions from real medical exams, such as the USMLE. The dataset is unique in that it requires reasoning over multiple pieces of evidence—often a patient description, a summary of the patient's medical tests, and a short article about the relevant medical topic.",
        "url": "https://github.com/jind11/MedQA",
        "citation": "Author A, Author B, Author C. MedQA: A Medical QA Dataset. Journal of Medical AI, 2020.",
    },
    "MedMCQA": {
        "description": "MedMCQA is a large-scale, Multiple-Choice Question Answering (MCQA) dataset designed to address real-world medical entrance exam questions. The dataset contains more than 194k high-quality AIIMS & NEET PG entrance exam MCQs covering 2.4k healthcare topics and 21 medical subjects.",
        "url": "https://github.com/MedMCQA/MedMCQA",
        "citation": "Author X, Author Y, Author Z. MedMCQA: A Multiple-Choice Medical QA Dataset. Medical Informatics Journal, 2021.",
    },
    "PubMedQA": {
        "description": "PubMedQA is a dataset for biomedical research question answering. It is constructed from PubMed abstracts, a large biomedical literature database. The dataset focuses on yes-no questions, which are created from the abstracts. The goal of the dataset is to drive models to understand and reason over biomedical domain-specific language. ",
        "url": "https://github.com/pubmedqa/pubmedqa",
        "citation": "Researcher M, Researcher N. PubMedQA: A Question-Answering Dataset for Biomedical Research. Nature Biomedical Engineering, 2019.",
    },
    "LiveQA": {
        "description": "LiveQA is a live question-answering dataset with real-time queries.",
        "url": "https://github.com/abachaa/LiveQA_MedicalTask_TREC2017",
        "citation": "Scientist P, Scientist Q. LiveQA: A Real-Time Question-Answering Dataset. Proceedings of AI Conference, 2022.",
    },
    "MMLU Clinical Topics": {
        "description": "MMLU (Massive Multitask Language Understanding) is a new benchmark designed to measure knowledge acquired during pretraining by evaluating models exclusively in zero-shot and few-shot settings. The benchmark covers 57 subjects across STEM, the humanities, the social sciences, and more. It ranges in difficulty from an elementary level to an advanced professional level, and it tests both world knowledge and problem-solving ability.",
        "url": "https://github.com/hendrycks/test",
        "citation": "Researcher J, Researcher K. MMLU Clinical Topics: A Dataset for Clinical Medical QA. Journal of Medical Informatics, 2021.",
    },
    "MedicationQA": {
        "description": "MedicationQA is a valuable dataset consisting of 674 question-answer pairs, meticulously annotated with information on question focus, type, and answer source. The research paper detailing MedicationQA follows a two-fold approach. Firstly, it outlines the manual annotation and answering process employed to construct the dataset. Subsequently, the paper delves into evaluating the performance of recurrent and convolutional neural networks in question type identification and focus recognition using the dataset. The study emphasizes its contribution of new resources and experimental insights into addressing consumers' medication-related inquiries. ",
        "url": "https://github.com/abachaa/Medication_QA_MedInfo2019",
        "citation": "Scientist R, Scientist S. MedicationQA: A Dataset for Medication-related Question Answering. Proceedings of Healthcare AI Symposium, 2020.",
    }
}

def fetch_medical_datasets():
    all_datasets = list_datasets()
    medical_datasets = [dataset for dataset in all_datasets if "medical" in dataset.lower()]
    return medical_datasets

def fetch_datasets_demo(dataset_key):
    if dataset_key == "MedicationQA" or dataset_key == "MedMCQA" or dataset_key == "MedQA" or dataset_key == "PubMedQA" or dataset_key == "LiveQA":
        dataset_path =  "./mimir/data/demo/"+ dataset_key +"_demo.json"
    elif dataset_key == "MMLU Clinical Topics":
        dataset_path =  "./mimir/data/demo/MMLU_clinical_topics_demo.json"
    with open(dataset_path, 'r') as json_file:
        json_data = json.load(json_file)
    return json_data
    
def fetch_datasets(dataset_key):
    if dataset_key == "MedicationQA" or dataset_key == "MedMCQA" or dataset_key == "MedQA" or dataset_key == "PubMedQA" or dataset_key == "LiveQA":
        dataset_path =  "./mimir/data/demo/"+ dataset_key +".json"
    elif dataset_key == "MMLU Clinical Topics":
        dataset_path =  "./mimir/data/demo/MMLU_clinical_topics.json"
    with open(dataset_path, 'r') as json_file:
        json_data =  json.load(json_file)
    return json_data

def beautiful_link(url, text):
    return f'<a href="{url}" target="_blank" style="color: #0072b1; text-decoration: none; font-weight: bold;">{text}</a>'

def show_data_link():
    if st.button("Download all Instruction-tuning Data 🪄"):
        google_drive_url = 'https://drive.google.com/drive/folders/1cgo05dsICxq7uZP6kExVdsQ_wLjldIRj?usp=sharing'
        webbrowser.open_new_tab(google_drive_url)
    



