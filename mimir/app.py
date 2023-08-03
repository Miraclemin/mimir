#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import multiprocessing
from multiprocessing import Process, Value, Queue
from hashlib import sha256
from multiprocessing import Manager, Pool
import streamlit as st
import re
from datasets import get_dataset_infos
from datasets.info import DatasetInfosDict
from pygments.formatters import HtmlFormatter
from utils import (
    get_dataset,
    get_dataset_confs,
    list_datasets,
    removeHyphen,
    renameDatasetColumn,
    render_features,
)
from conf.config import *
from chat_method.double_agent import *
from chat_method.mutil_agent import *
from chat_method.verify_construct import *
from role.role import *
# from chat_method.finetune import *

# DATASET_INFOS_CACHE_DIR = os.path.join(DEFAULT_PROMPTSOURCE_CACHE_HOME, "DATASET_INFOS")
# os.makedirs(DATASET_INFOS_CACHE_DIR, exist_ok=True)
# Python 3.8 switched the default start method from fork to spawn. OS X also has
# some issues related to fork, eee, e.g., https://github.com/bigscience-workshop/promptsource/issues/572
# so we make sure we always use spawn for consistency
PATTERN = r'[{,]("?\w+"?):\s*(".*?"|\[.*?\]|\{.*?\}|\d+\.?\d*)'

multiprocessing.set_start_method("spawn", force=True)

def get_infos(all_infos, d_name):
    d_name_bytes = d_name.encode("utf-8")
    d_name_hash = sha256(d_name_bytes)
    foldername = os.path.join(DATASET_INFOS_CACHE_DIR, d_name_hash.hexdigest())
    if os.path.isdir(foldername):
        infos_dict = DatasetInfosDict.from_directory(foldername)
    else:
        infos = get_dataset_infos(d_name)
        infos_dict = DatasetInfosDict(infos)
        os.makedirs(foldername)
        infos_dict.write_to_directory(foldername)
    all_infos[d_name] = infos_dict

select_options = ["Format 1", "Agent Talk", "Training a LLM"]
MODEL_SELECT_OPTION = ['LLaMA-7B', 'LLaMA-13B', 'Vicuna-7B', 'Vicuna-13B']
MODEL2HF_DCT = {'LLaMA-7B': 'decapoda-research/llama-7b-hf', 'LLaMA-13B': 'decapoda-research/llama-13b-hf',
                 'Vicuna-7B': 'lmsys/vicuna-7b-v1.3', 'Vicuna-13B': 'lmsys/vicuna-13b-v1.3'}
DATASET_SELCET_OPTION = ['alpaca 52k', 'ShareGPT']
DASET2HF_DCT = {'alpaca': 'tatsu-lab/alpaca', 'ShareGPT': 'RyokoAI/ShareGPT52K'}
side_bar_title_prefix = "Talky"

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
        options=select_options,
        index=0,
        key="mode_select",
    )
    st.sidebar.title(f"{side_bar_title_prefix} - {mode}")
    st.markdown(
        "<style>" + HtmlFormatter(style="friendly").get_style_defs(".highlight") + "</style>", unsafe_allow_html=True
    )
    if mode == "Format 1":
        st.title("Format 1")
        st.write("TO BE DEV")
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
                    num_epochs = num_epochs, learning_rate = 3e-4,
                    cutoff_len = cutoff_len, lora_r = lora_r,
                    lora_alpha = lora_alpha)
            
    if mode == "Agent Talk":
        topic_list = []
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
            print(topic_list)
            num_topic_file = len(topic_list)
        slider = st.sidebar.checkbox('Enable Exists Datasets')
        dataset_key = None
        if slider:
            dataset_key = st.sidebar.selectbox(
                "Dataset",
                dataset_list,
                key="dataset_select",
                index=ag_news_index,
                help="Select the dataset to work on.",
            )
        if dataset_key is not None:
            #
            # Check for subconfigurations (i.e. subsets)
            #
            configs = get_dataset_confs(dataset_key)
            conf_option = None
            if len(configs) > 0:
                conf_option = st.sidebar.selectbox("Subset", configs, index=0, format_func=lambda a: a.name)

            subset_name = str(conf_option.name) if conf_option else None
            try:
                dataset = get_dataset(dataset_key, subset_name)
            except OSError as e:
                st.error(
                    f"Some datasets are not handled automatically by `datasets` and require users to download the "
                    f"dataset manually. This applies to {dataset_key}{f'/{subset_name}' if subset_name is not None else ''}. "
                    f"\n\nPlease download the raw dataset to `~/.cache/promptsource/{dataset_key}{f'/{subset_name}' if subset_name is not None else ''}`. "
                    f"\n\nYou can choose another cache directory by overriding `PROMPTSOURCE_MANUAL_DATASET_DIR` environment "
                    f"variable and downloading raw dataset to `$PROMPTSOURCE_MANUAL_DATASET_DIR/{dataset_key}{f'/{subset_name}' if subset_name is not None else ''}`"
                    f"\n\nOriginal error:\n{str(e)}"
                )
                st.stop()

            splits = list(dataset.keys())
            index = 0
            if "train" in splits:
                index = splits.index("train")
            split = st.sidebar.selectbox("Split", splits, key="split_select", index=index)
            dataset = dataset[split]
            dataset = renameDatasetColumn(dataset)

            st.sidebar.subheader("Select Example")
            example_index = st.sidebar.slider("Select the example index", 0, len(dataset) - 1)
            example = dataset[example_index]
            example = removeHyphen(example)
            st.sidebar.write(example)

            st.sidebar.subheader("Dataset Schema")
            rendered_features = render_features(dataset.features)
            st.sidebar.write(rendered_features)
            st.header("Dataset: " + dataset_key + " " + (("/ " + conf_option.name) if conf_option else ""))

            # If we have a custom dataset change the source link to the hub
            split_dataset_key = dataset_key.split("/")
            possible_user = split_dataset_key[0]
            if len(split_dataset_key) > 1 and possible_user in INCLUDED_USERS:
                source_link = "https://huggingface.co/datasets/%s/blob/main/%s.py" % (
                    dataset_key,
                    split_dataset_key[-1],
                )
            else:
                source_link = "https://github.com/huggingface/datasets/blob/master/datasets/%s/%s.py" % (
                    dataset_key,
                    dataset_key,
                )
            st.markdown("*Homepage*: " + dataset.info.homepage + "\n\n*Dataset*: " + source_link)

            md = """
            %s
            """ % (
                dataset.info.description.replace("\\", "") if dataset_key else ""
            )
            st.markdown(md)

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
                topic_list = topic_list[::sample_index_step]
                st.markdown("*File Content:*")
                for topic_item in topic_list:
                    topic_str = "##### *Topic:* " + topic_item
                    st.markdown(topic_str)
                    st.markdown("***")
                    # st.markdown(f'<span style="color:#DAA528"> {topic_str} </span>', unsafe_allow_html=True)
            dialogue = ''
            st.write('\n')
            slider_advanced_setting = st.checkbox('Advanced Setting 🔧')
            picked_roles = []
            role_prompt = {}
            if slider_advanced_setting:
                Roles = Role('./talky/role/role.json')
                all_roles = Roles.all_roles_name
                role_prompt = Roles.all_roles
                # st.write(all_roles)
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
            setting_done = st.button('Begin to Talk Demo  🚀')
            verify_button = st.button('Begin to Verify 👾')

            if setting_done:
                if topic:
                    topic_list.append(topic)
                if topic_list:
                    if slider_advanced_setting and len(picked_roles) != 0:
                        progress_bar = st.progress(0.0)
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
                        chat_content = queue.get()
                        process.join()
                        st.write("Finished！")
                        # chat_content = mutil_agent(topic_list, index_list, picked_roles = picked_roles,
                            #role_prompt = role_prompt, MEMORY_LIMIT = configure["memory_limit"],
                            #asure = configure["azure"], max_rounds = max_rounds, max_input_token = max_input_token,
                            #user_temperature = user_temperature, ai_temperature=ai_temperature
                         #)
                    else:
                        progress_bar = st.progress(0.0)
                        process = Process(target=baize_demo, args=(queue,
                                                                   progress,
                                                                   topic_list,
                                                                   index_list,
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
                        chat_content = queue.get()
                        process.join()
                        st.write("Finished！")
                else:
                    st.write('Please Input The Topic Or Upload the Topic File')
        with col2:
            st.subheader("Generation")
            if setting_done:
                st.subheader("Talk Demo")
                # st.progress(_process)
                if len(chat_content):
                    for key, value in chat_content.items():
                        topic_str = "#### *Topic:* " + key
                        st.markdown(topic_str)
# <<<<<<< HEAD:talky/app.py
#                         lines = value.split('[Human]')
#                         dialogue_lst = []
#                         for line in lines:
#                             if len(line) > 0:
#                                 tmp_dct = {}
#                                 line_temp = line.split('[AI]')
#                                 tmp_dct['human'] = line_temp[0]
#                                 tmp_dct['ai'] = line_temp[1]
#                                 dialogue_lst.append(tmp_dct)
#                         for cnt, item in enumerate(dialogue_lst):
#                             human_rsp = item['human']
#                             ai_rsp = item['ai']
#                             st.markdown("#### *Human:*")
#                             st.markdown(f'<span style="color:#DAA520">{human_rsp}</span>', unsafe_allow_html=True)
#                             st.markdown("#### *AI:*")
#                             st.markdown(f'<span style="color:#00FF00">{ai_rsp}</span>', unsafe_allow_html=True)
#                             st.markdown("***")
                            
#                     st.markdown("***")
#                     st.session_state.Dialogue = dialogue_lst
# =======
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
                        # narration_after_verify = eval(narration_after_verify)
                        verify_lst.append({'human': human_rsp, 
                                    'ai': narration_after_verify})
                    except:
                        narration_after_verify = ai_rsp
                        verify_lst.append({'human': human_rsp, 'ai': ai_rsp})
                    
                st.session_state.VerifyDialogue = verify_lst

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
                        
    # state.sync()
                file_process = st.button('Begin to Process file ♻️')

if __name__ == "__main__":
    run_app()
