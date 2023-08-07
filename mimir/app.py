import multiprocessing
from multiprocessing import Process, Value, Queue
from hashlib import sha256
from multiprocessing import Manager, Pool, cpu_count
import streamlit as st
import pandas as pd
import re
from datasets import get_dataset_infos
from datasets.info import DatasetInfosDict
from pygments.formatters import HtmlFormatter
from conf.dataset_info_config import datasets_info,fetch_medical_datasets,fetch_datasets,fetch_datasets_demo,show_data_link
from utils import (
    process_double_agent,
    process_mutil_agent,
    save_dict_to_json
)
from conf.instruction_config import *
from conf.options_config import *
from conf.config import *
from chat_method.double_agent import *
from chat_method.mutil_agent import *
from chat_method.verify_construct import *
from role.role import *
# from finetune_method.finetune import *


multiprocessing.set_start_method("spawn", force=True)
side_bar_title_prefix = "Mimir"


def get_topic_list(dataset,dataset_key):
    local_topic_list = []  
    for item in dataset:
        question = item['Question']
        answer = item['Answer']
        if question and answer :
            topic = "Question:" + question +"\n" + "Answer:" + answer
        elif question and answer == None:
            topic = "Question:" + question
        local_topic_list.append(topic)
    return local_topic_list
        

def worker(results,topic_list,max_rounds,max_input_token,user_temperature,ai_temperature,api_key):
    """The worker function, invoked in a separate process."""
    chat_content,_ = baize_demo(
                        topic_list=topic_list, 
                        asure=True, 
                        max_rounds=max_rounds,
                        max_input_token=max_input_token, 
                        user_temperature=user_temperature, 
                        ai_temperature=ai_temperature, 
                        api_key = api_key
                    )
    results.append(chat_content)
     

def run_app():
    if 'Dialogue' not in st.session_state:
        st.session_state['Dialogue'] = []
    if 'Verify_dialogue' not in st.session_state:
        st.session_state['VerifyDialogue'] = []
    st.set_page_config(page_title="mimir", layout="wide")
    st.sidebar.image('./assets/logo.jpg')
    st.sidebar.markdown(
        "<center><a href='https://github.com/Miraclemin/mimir' style='font-size: 40px;'>Mimir\n\n</a></center>",
        unsafe_allow_html=True,
    )
    mode = st.sidebar.selectbox(
        label="DataGen Method",
        options=SELECT_OPTIONS,
        index=0,
        key="mode_select",
    )
    st.sidebar.title(f"{side_bar_title_prefix} - {mode}")

    flag_loop = 0
    api_key = st.sidebar.text_input("Enter your OpenAI API key:")
    if len(configure["open_ai_api_key"]) == 0 and api_key == "":
        st.write('Please Input Your OpenAI API Key Or Set In Config File!')
        flag_loop = 1
    while flag_loop:
        time.sleep(1)
    st.markdown(
        "<style>" + HtmlFormatter(style="friendly").get_style_defs(".highlight") + "</style>", unsafe_allow_html=True
    )
    processes = []
    # Create a manager object to manage shared data between processes
    manager = Manager()
    # Create a list managed by the Manager
    results = manager.list()
    if mode == "Training a LLM":
        col1, _, col2 = st.columns([12,1,20])
        chat_content = {}
        # 在第一列中放置第一个按钮
        with col1:
            st.subheader("Training Setting")
            batch_size = st.slider('Batch Size', 1, 512, 128)
            eval_steps = st.slider('Eval Steps', 0, 1000, 10)
            save_steps = st.slider('Save Steps', 0, 1000, 10)
            cutoff_len = st.slider('Cutoff Len', 1, 2048, 512)
            num_epochs = st.slider('Num Epochs', 1, 10, 3)
            lora_r = st.slider('Lora Rank', 1, 32, 8)
            lora_alpha = st.slider('Lora Alpha', 1, 32, 16)
            model = st.sidebar.selectbox(
                label="Base Model Selection",
                options=MODEL_SELECT_OPTION,
                index=0,
                key="model_select",
            )
            dataset = st.sidebar.selectbox(
                label="Dataset Selection",
                options=DATASET_SELCET_OPTION,
                index=0,
                key="dataset_select",
            )
            base_model = MODEL2HF_DCT[model]
            base_dataset = DASET2HF_DCT[dataset]
            train_button = st.button('Begin to Train 👽')
            if train_button:
                train(base_model = base_model, data_path = base_dataset,
                      output_dir =  "./output/saved_model", eval_steps = eval_steps, 
                      save_steps = save_steps, batch_size = batch_size,
                      num_epochs = num_epochs, learning_rate = 5e-5,
                      cutoff_len = cutoff_len, lora_r = lora_r,
                      lora_alpha = lora_alpha)
        with col2:
            st.subheader("Shell to train YOUR OWN LLM!!!")
            shell_button = st.button('Begin to Generate shell command👽')
            if shell_button:
                txt = st.text_area('Shell command', 
                                   f"python finetune.py --base_model {base_model} --data_path {base_dataset} --output_dir ./output/saved_model --batch_size {batch_size} --num_epochs {num_epochs} --learning_rate 5e-5 --cutoff_len {cutoff_len} --val_set_size 2000 --lora_r {lora_r} --lora_alpha {lora_alpha} --lora_dropout 0.05 --lora_target_modules '[q_proj,v_proj]' --train_on_inputs")

    if mode == "Medical Dataset":
        # fetch current dataset
        # medical_datasets = fetch_medical_datasets()
        slider = st.sidebar.checkbox('Tune single dataset customly')
        
        dataset_list = ["MedQA","MedMCQA","PubMedQA","MMLU Clinical Topics","MedicationQA","LiveQA"]
        st.title("Medical Dataset")
        dataset_key = None
        if slider:
            dataset_key = st.sidebar.selectbox(
                "Dataset",
                dataset_list,
                key="dataset_select",
                index=0,
                help="Select the dataset to work on.",
            )
        else:
            show_data_link()
            st.subheader("Dataset Setting 💡")
            selected_options = st.multiselect(
        "Select one or more medical datasets",
        dataset_list)
            st.subheader("Talk Setting 🔦")
            max_rounds = st.slider('Max Rounds', 0, 100, 1)
            max_input_token = st.slider('Max Input Tokens', 0, 3000, 100)
            user_temperature = st.slider('Human Temperature', 0.0, 1.0, 0.1)
            ai_temperature = st.slider('AI Temperature', 0.0, 1.0, 0.1)
            setting_done = st.button("Begin to process your choosed dataset 🚀 ",)
            if setting_done:
                if selected_options:
                    topic_list = []
                    for ds_key in selected_options:
                        if ds_key == "MedicationQA" or ds_key == "MedMCQA" or ds_key == "MedQA" or ds_key == "PubMedQA" or ds_key == "LiveQA":
                            dataset_path = "./mimir/data/"+ ds_key +".json"
                        elif ds_key == "MMLU Clinical Topics":
                            dataset_path = "./mimir/data/MMLU_clinical_topics.json"
                        with open(dataset_path, 'r') as json_file:
                            json_data = json.load(json_file)
                        temp_topic = get_topic_list(json_data,ds_key)
                        topic_list += temp_topic
                else:
                    st.write("Please select at least one dataset before producing instructions ⏱")        
                    
                num_entries = len(json_data)
                process_num = cpu_count()
                
                # We split the dataset into chunks and process each chunk in a separate process:
                chunk_size = num_entries // process_num  
                chunks = [json_data[i:i + chunk_size] for i in range(0, len(json_data), chunk_size)]

                st.write("There are "+ str(num_entries)+ " data entries in the your chosen dataset 🌟 \n ")
                if topic_list:
                    
                    # Create processess
                    for chunk in chunks:
                        p = Process(target=worker, args=(results,chunk,max_rounds,
                        max_input_token,user_temperature,ai_temperature, api_key))
                        processes.append(p)
                    
                    # Start the processes
                    for p in processes:
                        p.start()
                        # print("Currentky results are: " + str(results.count))

                    # Ensure all processes have finished execution
                    for p in processes:
                        p.join()
                        
                else:
                    st.write('Please Input The Topic Or Upload the Topic File')
                
            
        if dataset_key is not None:
            dataset_demo = fetch_datasets_demo(dataset_key)
            # Parse the JSON string to obtain a Python data structure (e.g., list)
            st.header(dataset_key+" 📜")
            ds_description = datasets_info[dataset_key]["description"]
            ds_url = datasets_info[dataset_key]["url"]
            st.write(ds_description)
            st.markdown("Repo: %s" % ds_url)
            st.caption("Dataset Viewer")
            st.dataframe(pd.DataFrame(dataset_demo))
            file_contents = pd.DataFrame(dataset_demo)
            st.download_button(label="Download instruction data processed from "+dataset_key+ " 🔥", data=file_contents.to_csv(), file_name="processed_file.csv")
            
            st.header("Dataset Tuning 🔧")
            st.subheader("Talk Setting")
            max_rounds = st.slider('Max Rounds', 0, 100, 1)
            max_input_token = st.slider('Max Input Tokens', 0, 3000, 100)
            user_temperature = st.slider('Human Temperature', 0.0, 1.0, 0.1)
            ai_temperature = st.slider('AI Temperature', 0.0, 1.0, 0.1)
            # api_key = st.text_input("Enter your OpenAI API key:")
            st.write('\n')
            setting_done = st.button("Begin to process "+ dataset_key+ " 🚀 ",)
            if setting_done:
                json_data = fetch_datasets(dataset_key)
                topic_list = get_topic_list(json_data,dataset_key)
            
                num_entries = len(json_data)
                process_num = cpu_count()
                
                # We split the dataset into chunks and process each chunk in a separate process:
                chunk_size = num_entries // process_num  
                chunks = [json_data[i:i + chunk_size] for i in range(0, len(json_data), chunk_size)]

                st.write("There are "+ str(num_entries)+ " data entries in the "+ dataset_key + " 🌟 \n ")
                if topic_list:
                    
                    # Create processes
                    for chunk in chunks:
                        p = Process(target=worker, args=(results,chunk,max_rounds,
                        max_input_token,user_temperature,ai_temperature, api_key))
                        processes.append(p)
                    
                    # Start the processes
                    for p in processes:
                        p.start()
                        # print("Currentky results are: " + str(results.count))

                    # Ensure all processes have finished execution
                    for p in processes:
                        p.join()
                        
                else:
                    st.write('Please Input The Topic Or Upload the Topic File')
    if mode == "Agent Talk":
        topic_list = []
        topic_file_list = []
        index_list = []
        ### 用户自定义topic
        topic = st.sidebar.text_input('Topic Input：')
        if topic:
            topic_list.append(topic)
        ### 用户上传topic 文件
        num_topic_file = 0
        uploaded_file = st.sidebar.file_uploader("Upload a Topic file")
        # 如果用户上传了文件，将文件内容作为对话主题
        if uploaded_file is not None:
            # 假设文件是文本文件，直接读取文件内容作为对话主题
            topic_file = uploaded_file.read().decode().split('\n')
            for item in topic_file:
                if item:
                    topic_list.append(item)
            # print(topic_list)
            num_topic_file = len(topic_list)

        # 将页面分割为两列
        col1, _, col2 = st.columns([12,1,12])
        chat_content = {}
        # 在第一列中放置第一个按钮
        progress = Value('d', 0.0)
        place_text = st.text("")
        queue = Queue()
        with col1:
            st.subheader("Talk Setting")
            max_rounds = st.slider('Max Rounds', 0, 10, 1)
            max_input_token = st.slider('Max Input Tokens', 0, 3000, 100)
            user_temperature = st.slider('Human Temperature', 0.0, 1.0, 0.1)
            ai_temperature = st.slider('AI Temperature', 0.0, 1.0, 0.1)
            if uploaded_file:
                sample_index_step = st.slider('Sample Step', 1, num_topic_file, int(num_topic_file/2))
                topic_file_list = topic_list[:]
                topic_list = topic_list[::sample_index_step]
                st.markdown("*File Content:*")
                for topic_item in topic_list:
                    topic_str = "##### *Topic:* " + topic_item
                    st.markdown(topic_str)
                    st.markdown("***")
            st.write('\n')
            slider_advanced_setting = st.checkbox('Advanced Setting 🔧')
            picked_roles = []
            role_prompt = {}
            if slider_advanced_setting:
                Roles = Role('./mimir/role/role.json')
                all_roles = Roles.all_roles_name
                role_prompt = Roles.all_roles
                num_agents = st.slider('Number of Agents', 2, len(all_roles), 1)
                for index in range(num_agents):
                    variable_name = "Agent_" + str(index)
                    globals()[variable_name] = st.selectbox(
                                                    " ".join(variable_name.split("_")),
                                                    all_roles,
                                                    key="roles_select" + variable_name,
                                                    index=0,
                                                    help="Select the roles to work on.",
                    )
                    picked_roles.append(globals()[variable_name])
                    st.markdown(f'<span style="font-size: 15px; color: Green;"><i><b>{Roles.all_roles[globals()[variable_name]]}</i></b></span>', unsafe_allow_html=True)
            st.write('\n')
            col1_but, col2_but, col3_but = st.columns([3, 3, 3])
            with col1_but:
                setting_done = st.button('Begin to Talk Demo  🚀')
            with col2_but:
                verify_button = st.button('Begin to Verify 👾')
            with col3_but:
                file_process = st.button('Begin to Process file ♻️')
            if setting_done:
                if topic_list:
                    progress_bar = st.progress(0.0)
                    if slider_advanced_setting and len(picked_roles) != 0:
                        process = Process(target=mutil_agent, args=(queue,
                                                                    progress,
                                                                    topic_list,
                                                                    index_list,
                                                                    picked_roles,
                                                                    role_prompt,
                                                                    configure["memory_limit"],
                                                                    configure["azure"],
                                                                    max_rounds,
                                                                    max_input_token,
                                                                    user_temperature,
                                                                    ai_temperature))
                        process.start()
                        while process.is_alive():
                            if progress.value >= 1:
                                break
                            progress_bar.progress(progress.value)
                            place_text.text(f"Progress: {progress.value}%")
                        progress_bar.progress(1.0)
                        place_text.text("Progress: 1.00 %")
                        chat_content = queue.get()
                        process.join()
                        st.write("Finished！")
                    else:

                        process = Process(target=baize_demo, args=(queue,
                                                                   progress,
                                                                   topic_list,
                                                                   index_list,
                                                                   configure["azure"],
                                                                   max_rounds,
                                                                   max_input_token,
                                                                   user_temperature,
                                                                   ai_temperature,
                                                                   api_key))
                        process.start()
                        while process.is_alive():
                            if progress.value >= 1:
                                break
                            progress_bar.progress(progress.value)
                            place_text.text(f"Progress: {progress.value}%")

                        progress_bar.progress(1.0)
                        place_text.text("Progress: 1.00 %")
                        chat_content = queue.get()
                        process.join()
                        st.write("Finished！")
                else:
                    st.write('Please Input The Topic Or Upload the Topic File')
        with col2:
            if setting_done:
                st.subheader("Talk Demo")
                if len(chat_content):
                    for key, value in chat_content.items():
                        topic_str = "#### *Topic:* " + key
                        st.markdown(topic_str)

                        if slider_advanced_setting and len(picked_roles) != 0:
                            # Processing the dialogue
                                for response in value:
                                    name = response.split(':')[0]
                                    res_content = " ".join(response.split(':')[1:])
                                    st.markdown(f"#### *{name}:*")
                                    st.markdown(f'<span style="color:#DAA520">{res_content}</span>',
                                                unsafe_allow_html=True)
                        else:
                            lines = value.split('[Human]')
                            dialogue_lst = []
                            for line in lines:
                                if len(line) > 0:
                                    tmp_dct = {}
                                    line_temp = line.split('[AI]')
                                    tmp_dct['human'] = line_temp[0]
                                    tmp_dct['ai'] = line_temp[1]
                                    dialogue_lst.append(tmp_dct)
                            for cnt, item in enumerate(dialogue_lst):
                                human_rsp = item['human']
                                ai_rsp = item['ai']
                                st.markdown("#### *Human:*")
                                st.markdown(f'<span style="color:#DAA520">{human_rsp}</span>', unsafe_allow_html=True)
                                st.markdown("#### *AI:*")
                                st.markdown(f'<span style="color:#00FF00">{ai_rsp}</span>', unsafe_allow_html=True)
                                st.markdown("***")
                            st.session_state.Dialogue = dialogue_lst
                else:
                    st.write('Error in the talking generation')
                st.write('\n')
            if verify_button:
                verify_lst = []
                for cnt, item in enumerate(st.session_state.Dialogue):
                    human_rsp = item['human']
                    ai_rsp = item['ai']
                    try:
                        narration_after_verify = verify(human_rsp, ai_rsp)
                        verify_lst.append({'human': human_rsp, 
                                    'ai': narration_after_verify})
                    except:
                        narration_after_verify = ai_rsp
                        verify_lst.append({'human': human_rsp, 'ai': ai_rsp})
                st.session_state.VerifyDialogue = verify_lst


            progress = Value('d', 0.0)
            # place_text = st.text("")
            place_text.text("")
            if uploaded_file and file_process and len(topic_file_list) != 0:
                if slider_advanced_setting and len(picked_roles) != 0:
                    # progress_bar = st.progress(0.0)
                    progress_bar.progress(0.0)
                    process = Process(target=mutil_agent, args=(queue,
                                                                progress,
                                                                topic_file_list,
                                                                index_list,
                                                                picked_roles,
                                                                role_prompt,
                                                                configure["memory_limit"],
                                                                configure["azure"],
                                                                max_rounds,
                                                                max_input_token,
                                                                user_temperature,
                                                                ai_temperature))
                    process.start()
                    while process.is_alive():
                        if progress.value >= 1:
                            break
                        progress_bar.progress(progress.value)
                        place_text.text(f"Progress: {progress.value}%")
                    progress_bar.progress(1.0)
                    place_text.text("Progress: 1.00 %")
                    chat_content = queue.get()
                    process.join()
                    # st.write("Finished！")
                    ### multiagent processing
                    date_download = []
                    date_download = process_mutil_agent(chat_content, picked_roles)
                    save_dict_to_json(date_download, 'data.json')
                    with open('data.json', 'r') as f:
                        data = f.read()
                    st.download_button(label='Click to Download', data=data, file_name='data.json',
                                       mime='application/json')
                else:
                    # progress_bar = st.progress(0.0)
                    progress_bar.progress(0.0)
                    process = Process(target=baize_demo, args=(queue,
                                                               progress,
                                                               topic_file_list,
                                                               index_list,
                                                               configure["azure"],
                                                               max_rounds,
                                                               max_input_token,
                                                               user_temperature,
                                                               ai_temperature,
                                                               api_key))
                    process.start()
                    while process.is_alive():
                        if progress.value >= 1:
                            break
                        progress_bar.progress(progress.value)
                        place_text.text(f"Progress: {progress.value}%")

                    progress_bar.progress(1.0)
                    place_text.text("Progress: 1.00 %")
                    chat_content = queue.get()
                    process.join()
                    # st.write("Finished！")
                    #### double agent processing
                    date_download = []
                    date_download = process_double_agent(chat_content)
                    # print(date_download)
                    save_dict_to_json(date_download, 'data.json')
                    with open('data.json', 'r') as f:
                        data = f.read()
                    st.download_button(label='Click to Download', data=data, file_name='data.json',
                                       mime='application/json')
            else:
                st.write('Please Upload the Topic File!')
        # Begin to view
        if verify_button:
            for cnt, item in enumerate(st.session_state.Dialogue):
                human_rsp = item['human']
                ai_rsp = item['ai']
                with col2:
                    st.markdown("#### *Human:*")
                    st.markdown(f'<span style="color:#DAA520">{human_rsp}</span>', unsafe_allow_html=True)
                    st.markdown("#### *AI:*")
                    st.markdown(f'<span style="color:#00FF00">{ai_rsp}</span>', unsafe_allow_html=True)
                    st.markdown("***")
                    item_verified = st.session_state.VerifyDialogue[cnt]
                    ai_rsp_verified = item_verified['ai']
                    st.markdown("#### *💡Verifacation:*")
                    st.markdown(f'<span style="color:#9ACD32">{ai_rsp_verified}</span>', unsafe_allow_html=True)
                    st.markdown("***")


if __name__ == "__main__":
    run_app()