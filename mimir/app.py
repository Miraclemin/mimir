import multiprocessing
from multiprocessing import Process, Value, Queue
from hashlib import sha256
from multiprocessing import Manager, Pool
import streamlit as st
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
from chat_method.baize import *
from chat_method.mutil_agent import *
from role.role import *

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

def get_topic_list(dataset,dataset_key):
    local_topic_list = []  
    for item in dataset:
        print("item:",item)
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
                        topic_list=topic_list, asure=True, max_rounds=max_rounds,
                        max_input_token=max_input_token, user_temperature=user_temperature, ai_temperature=ai_temperature, api_key = api_key
                    )
    results.append(chat_content)
     


select_options = ["Medical Dataset", "Agent Talk"]
side_bar_title_prefix = "Mimir"

def run_app():
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
    processes = []
    # Create a manager object to manage shared data between processes
    manager = Manager()
    # Create a list managed by the Manager
    results = manager.list()
    
    if mode == "Medical Dataset":
        #### exist datasets
        #dataset_list = list_datasets()
        #ag_news_index = dataset_list.index("quora")
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
            st.subheader("Dataset Setting 💡")
            selected_options = st.multiselect(
        "Select one or more medical datasets",
        dataset_list)
            st.subheader("Talk Setting 🔦")
            max_rounds = st.slider('Max Rounds', 0, 100, 1)
            max_input_token = st.slider('Max Input Tokens', 0, 3000, 100)
            user_temperature = st.slider('Human Temperature', 0.0, 1.0, 0.1)
            ai_temperature = st.slider('AI Temperature', 0.0, 1.0, 0.1)
            api_key = st.text_input("Enter your OpenAI API key:")
            setting_done = st.button("Begin to process your choosed dataset 🚀 ",)
            if setting_done:
                if selected_options:
                    topic_list = []
                    for ds_key in selected_options:
                        if ds_key == "MedicationQA" or ds_key == "MedMCQA" or ds_key == "MedQA" or ds_key == "PubMedQA" or ds_key == "LiveQA":
                            dataset_path = "./talky/data/"+ ds_key +".json"
                        elif ds_key == "MMLU Clinical Topics":
                            dataset_path = "./talky/data/MMLU_clinical_topics.json"
                        with open(dataset_path, 'r') as json_file:
                            json_data = json.load(json_file)
                        temp_topic = get_topic_list(json_data,ds_key)
                        topic_list += temp_topic
                else:
                    st.write("Please select at least one dataset before producing instructions ⏱")        
                    
                print(len(topic_list))
            
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
                        max_input_token,user_temperature,ai_temperature,api_key))
                        processes.append(p)
                    
                    # Start the processes
                    for p in processes:
                        p.start()
                        print("Currentky results are: " + str(results.count))

                    # Ensure all processes have finished execution
                    for p in processes:
                        p.join()
                        
                else:
                    st.write('Please Input The Topic Or Upload the Topic File')
                
            
        if dataset_key is not None:
            if dataset_key == "MedicationQA" or dataset_key == "MedMCQA" or dataset_key == "MedQA" or dataset_key == "PubMedQA" or dataset_key == "LiveQA":
                dataset_path = "./talky/data/"+ dataset_key +".json"
            elif dataset_key == "MMLU Clinical Topics":
                dataset_path = "./talky/data/MMLU_clinical_topics.json"
            # elif :
            #     dataset_path = "./talky/data/"+dataset_key+".json"
            with open(dataset_path, 'r') as json_file:
                json_data = json.load(json_file)
            # Parse the JSON string to obtain a Python data structure (e.g., list)
            
            print(json_data[0])
                        

            
            st.header(dataset_key+" 📜")
            ds_description = datasets_info[dataset_key]["description"]
            ds_url = datasets_info[dataset_key]["url"]
            st.write(ds_description)
            st.markdown("Repo: %s" % ds_url)
            st.caption("Dataset Viewer")
            st.dataframe(pd.DataFrame(json_data).head(50))
            file_contents = pd.DataFrame(json_data)
            st.download_button(label="Download instruction data processed from "+dataset_key+ " 🔥", data=file_contents.to_csv(), file_name="processed_file.csv")
            
            st.header("Dataset Tuning 🔧")
            st.subheader("Talk Setting")
            max_rounds = st.slider('Max Rounds', 0, 100, 1)
            max_input_token = st.slider('Max Input Tokens', 0, 3000, 100)
            user_temperature = st.slider('Human Temperature', 0.0, 1.0, 0.1)
            ai_temperature = st.slider('AI Temperature', 0.0, 1.0, 0.1)
            api_key = st.text_input("Enter your OpenAI API key:")
            dialogue = ''
            st.write('\n')
            setting_done = st.button("Begin to process "+ dataset_key+ " 🚀 ",)
            if setting_done:
                topic_list = get_topic_list(json_data,dataset_key)
                print(len(topic_list))
            
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
                        max_input_token,user_temperature,ai_temperature,api_key))
                        processes.append(p)
                    
                    # Start the processes
                    for p in processes:
                        p.start()
                        print("Currentky results are: " + str(results.count))

                    # Ensure all processes have finished execution
                    for p in processes:
                        p.join()
                        
                else:
                    st.write('Please Input The Topic Or Upload the Topic File')
    if mode == "Format 2":
        st.title("Format 2")
        st.write("TO BE DEV")
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
            if setting_done:
                st.subheader("Talk Demo")
                # st.progress(_process)
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
                            for line in lines:
                                if len(line) > 0:
                                    line_temp = line.split('[AI]')
                                    st.markdown("#### *Human:*")
                                    st.markdown(f'<span style="color:#DAA520">{line_temp[0]}</span>', unsafe_allow_html=True)
                                    st.markdown("#### *AI:*")
                                    st.markdown(f'<span style="color:#00FF00">{line_temp[1]}</span>', unsafe_allow_html=True)
                                    st.markdown("***")
                else:
                    st.write('Error in the talking generation')
                st.write('\n')
                file_process = st.button('Begin to Process file ♻️')

if __name__ == "__main__":
    run_app()
