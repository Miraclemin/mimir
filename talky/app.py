import multiprocessing
import functools
from hashlib import sha256
from multiprocessing import Manager, Pool
import streamlit as st
from datasets import get_dataset_infos
from datasets.info import DatasetInfosDict
from pygments.formatters import HtmlFormatter
# from talky import DEFAULT_PROMPTSOURCE_CACHE_HOME
from session import _get_state
from utils import (
    get_dataset,
    get_dataset_confs,
    list_datasets,
    removeHyphen,
    renameDatasetColumn,
    render_features,
)
from chat_method.baize import *

# DATASET_INFOS_CACHE_DIR = os.path.join(DEFAULT_PROMPTSOURCE_CACHE_HOME, "DATASET_INFOS")
# os.makedirs(DATASET_INFOS_CACHE_DIR, exist_ok=True)
# Python 3.8 switched the default start method from fork to spawn. OS X also has
# some issues related to fork, eee, e.g., https://github.com/bigscience-workshop/promptsource/issues/572
# so we make sure we always use spawn for consistency
multiprocessing.set_start_method("spawn", force=True)

def get_infos(all_infos, d_name):
    """
    Wrapper for mutliprocess-loading of dataset infos

    :param all_infos: multiprocess-safe dictionary
    :param d_name: dataset name
    """
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


select_options = ["Format 1", "Agent Talk", "Format 2"]
side_bar_title_prefix = "Talky"

# #
# # Cache functions
# #
# get_dataset = st.cache(allow_output_mutation=True)(get_dataset)
# get_dataset_confs = st.cache(get_dataset_confs)
# list_datasets = st.cache(list_datasets)

def run_app():
    state = _get_state()
    # Download dataset infos (multiprocessing download)
    # manager = Manager()
    # all_infos = manager.dict()
    # all_datasets = list(set([t[0] for t in template_collection.keys]))
    # pool = Pool(processes=multiprocessing.cpu_count())
    # pool.map(functools.partial(get_infos, all_infos), all_datasets)
    # pool.close()
    # pool.join()
    st.set_page_config(page_title="Talky", layout="wide")
    st.sidebar.image('./assets/logo.jpeg')
    st.sidebar.markdown(
        "<center><a href='https://github.com/' style='font-size: 40px;'>Talky\n\n</a></center>",
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
        #### exist datasets
        #dataset_list = list_datasets()
        #ag_news_index = dataset_list.index("quora")
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
        col1, _, col2 = st.beta_columns([12,1,12])
        chat_content = {}
        # 在第一列中放置第一个按钮
        with col1:
            st.subheader("Talk Setting")
            max_rounds = st.slider('Max Rounds', 0, 100, 1)
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
            setting_done = st.button('Begin to Talk Demo  🚀')
            if setting_done:
                if topic:
                    topic_list.append(topic)
                if topic_list:
                    chat_content, total_tokens = baize_demo(
                        topic_list, index_list, asure=True, max_rounds=max_rounds,
                        max_input_token=max_input_token, user_temperature=user_temperature, ai_temperature=ai_temperature
                    )
                else:
                    st.write('Please Input The Topic Or Upload the Topic File')
        with col2:
            if setting_done:
                st.subheader("Talk Demo")
                if len(chat_content):
                    for key, value in chat_content.items():
                        topic_str = "#### *Topic:* " + key
                        st.markdown(topic_str)
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
    state.sync()

if __name__ == "__main__":
    run_app()
