SELECT_OPTIONS = ["Medical Dataset", "Agent Talk", "Training a LLM"]
MODEL_SELECT_OPTION = ['LLaMA-7B', 'LLaMA-13B', 'Vicuna-7B', 'Vicuna-13B']
MODEL2HF_DCT = {'LLaMA-7B': 'decapoda-research/llama-7b-hf', 'LLaMA-13B': 'decapoda-research/llama-13b-hf',
                 'Vicuna-7B': 'lmsys/vicuna-7b-v1.3', 'Vicuna-13B': 'lmsys/vicuna-13b-v1.3'}
DATASET_SELCET_OPTION = ['alpaca 52k', 'ShareGPT']
DASET2HF_DCT = {'alpaca 52k': 'tatsu-lab/alpaca', 'ShareGPT': 'RyokoAI/ShareGPT52K'}
PATTERN = r'[{,]("?\w+"?):\s*(".*?"|\[.*?\]|\{.*?\}|\d+\.?\d*)'
