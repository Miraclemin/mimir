# Mimir

<p align="center">
  <img src="assets/logo.jpg" width="800"/>
</p>

*In Norse mythology, Mímisbrunnr, the Well of Wisdom, symbolizes the idea that models can draw knowledge from various domains and generate specialized datas for different fields, much like how Odin gained wisdom from the Well of Mímir.*

**Mimir is a medical domain multi-turn dialogue data generation tool based on ChatGPT. It supports multi-agent dialogues, allowing users to generate data for instruction tuning using publicly available medical domain datasets. Users can also upload their own knowledge documents to generate dialogue datasets based on those documents. Mimir supports dialogue verification, where each generated data is double-checked to ensure the accuracy of knowledge. Additionally, it supports fine-tuning, enabling users to deploy the platform on their own machines or utilize our provided fine-tuning scripts for training.**

## Deploy
Download processed data from: https://drive.google.com/file/d/1QcbArUZHhKWtvIakIoPbhBaUoPrTsljK/view?usp=drive_link and place the JSON files of this data in the './data' directory.

```bash
1. Download the repo
2. pip install -r requirements.txt
3. streamlit run mimir/app.py
```

## How to generate data?

## How to generate data based on own data file?

## How to fineturn using our data?

## Citation
If you find Mimir useful, please cite the following reference:
```bibtex

```
