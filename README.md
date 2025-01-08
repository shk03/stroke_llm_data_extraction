# Data Extraction from Free-Text Stroke CT Reports Using GPT-4o and Llama-3.3-70B: The Impact of Annotation Guidelines 

## Background
CT reports in cases of suspected acute stroke contain valuable data with great utility beyond clinical care, enabling epidemiological studies, training of machine learning algorithms, and national registries. In recent years, large language models (LLMs) have demonstrated potential in automating data mining from radiology reports. Yet, the role of annotation guidelines on LLM-based data extraction has not been investigated.

## Study Design
In this study, performance of GPT-4o and Llama-3.3-70B in extracting ten imaging findings from stroke CT reports was assessed in two datasets from a single academic stroke center. Dataset A (n = 200) was an artificial cohort including a variety of pathological findings, whereas Dataset B (n = 100) was a consecutive cohort. Initially, a comprehensive annotation guideline was designed based on a thorough review of cases with inter-annotator disagreements in dataset A. For each LLM, data extraction was performed under two conditions – with the annotation guideline included in the prompt and without it. 

<img width="452" alt="image" src="https://github.com/user-attachments/assets/1d46fd7f-d0be-4551-8b98-db2d78348828" />

## Results
Overall, GPT-4o consistently demonstrated superior performance over Llama-3.3-70B under identical conditions, with overall precision ranging from 0.83 to 0.95 for GPT-4o and from 0.65 to 0.86 for Llama-3.3-70B. Across both models and both datasets, higher precision rates were observed in the presence of the annotation guideline, while recall rates largely remained stable. In dataset B, overall precision of GPT-4o and Llama-3-70B improved from 0.83 to 0.95 and from 0.87 to 0.94, respectively. Overall classification performance with and without annotation guideline was significantly different in all four dataset-model pairs (e.g. dataset B/GPT-4o: p = 0.006; dataset B / Llama-3.3: p = 0.001). 

<img width="452" alt="image" src="https://github.com/user-attachments/assets/b163d021-d58e-4c3e-b6ac-375ba375379a" />

## Files contained in this repo

To allow for the full reproducibility of our study, we publish detailed model links (below), our annotation guideline, template, and scripts for executing LLM queries.

- template.json: list of data variables to be extracted
- annotation-guideline.txt: final, comprehensive annotation guideline with instructions on how to deal with potentially ambiguous cases
- gpt-4o-data-extraction.py: script for running GPT-4o via OpenAI's API
- llama-3.3-data-extraction.py: script for locally running Llama-3.3-70B

## LLMs used in our study
- GPT-4o: gpt-4o-2024-08-06 (https://platform.openai.com/docs/models#gpt-4o)
- Llama-3-70B (https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct!)

Please note that Llama-3-70B was quantized (GGUF format) and used with llama_cpp_python v. 0.2.89.

## FAQ
_What GPU do I need?_
As a rule of thumb, the model file size (.gguf) should be 1-2 GB smaller than your VRAM. Also, we currently fully offload models to the GPU's VRAM ("n_gpu_layers=-1" in the Llama() constructor call). If you change this to only partly offload to the GPU's memory, you can also load larger models / at higher quantization. However, this comes at a (relevant) speed penalty.


