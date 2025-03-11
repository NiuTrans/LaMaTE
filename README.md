# Beyond Decoder-only: Large Language Models Can be Good Encoders for Machine Translation

<p align="center">
  <a href="https://arxiv.org/abs/2503.06594" alt="paper"><img src="https://img.shields.io/badge/Paper-LaMaTE-blue?logo=arxiv&logoColor=white"/></a>
  <a href="https://huggingface.co/NiuTrans/LaMaTE" alt="Model"><img src="https://img.shields.io/badge/Model-LaMaTE-yellow?logo=huggingface"/></a>
  <a href="https://huggingface.co/datasets/NiuTrans/ComMT" alt="Dataset"><img src="https://img.shields.io/badge/Dataset-ComMT-yellow?logo=huggingface"/></a>
  <a href="https://github.com/NiuTrans" alt="NiuTrans"><img src="https://img.shields.io/badge/NiuTrans-blue"/></a>
  <a href="http://team.neu.edu.cn/NEUNLPLab/zh_CN/index.htm" alt="NEUNLP"><img src="https://img.shields.io/badge/NEUNLP-blue"/></a>
</p>


<div align="center">
<p align="center" dir="auto">

• 📄 [Introduction](#-introduction) 
• 🤗 [Model and Dataset](#-model-and-dataset)
• 🚀 [A Quick Start](#-a-quick-start)
</p>
<p align="center" dir="auto">

• 🔥 [Training](#-training) 
• ⚡ [Inference](#-inference) 
• 📊 [Evaluation](#-evaluation)
</p>
</div>

# 📄 Introduction
LaMaTE is a high-performance and efficient translation model that utilizes large language models(LLMs) as machine translation(MT) encoders, paired with lightweight decoders. 
The model integrates an adapter to bridge LLM representations with the decoder, employing a two-stage training strategy to enhance performance and efficiency.

**Key Features of LaMaTE**
- Enhanced Efficiency: Offers 2.4× to 6.5× faster decoding speeds.
- Reduced Memory Usage: Reduces KV cache memory consumption by 75%.
- Competitive Performance: Exhibits robust performance across diverse translation tasks.

ComMT is a comprehensive dataset suite designed to support the development and evaluation of universal translation models. 
It includes diverse translation-related tasks, providing a well-curated data resource for training and testing LLM-based machine translation systems.


# 🤗 Model and Dataset
We have made the following resources available:

| Resource         | Description                                         | Link                                                      |
|------------------|-----------------------------------------------------|-----------------------------------------------------------|
| LaMaTE    | The LaMaTE model, developed using Llama-3-8B	  | [🤗NiuTrans/LaMaTE](https://huggingface.co/NiuTrans/LaMaTE) |
| ComMT    | Dataset suite, includes 239k high-quality, diverse SFT data	  | [🤗NiuTrans/ComMT](https://huggingface.co/datasets/NiuTrans/ComMT) |


# 🚀 A Quick Start
**Note:** Our implementation is developed with transformers v4.39.2. 
We recommend installing this version for best compatibility.

To deploy LaMaTE, utilize the ```from_pretrained()``` method followed by the ```generate()``` method for immediate use:

```python
from modeling_llama_seq2seq import LlamaCrossAttentionEncDec
from transformers import AutoTokenizer, AutoConfig

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
model = LlamaCrossAttentionEncDec.from_pretrained(model_name_or_path, config=config)

prompt = "Translate the following text from English into Chinese.\nEnglish: The harder you work at it, the more progress you will make.\nChinese: ",
input_ids = tokenizer(prompt, return_tensors="pt")
outputs_tokenized = model.generate(
    **input_ids,
    num_beams=5,
    do_sample=False
)
outputs = tokenizer.batch_decode(outputs_tokenized, skip_special_tokens=True)
print(outputs) 
```

The prompt for general/doc/domain translation tasks:
```
"Translate the following text from {src_lang} into {tgt_lang}.\n{src_lang}: {src}\n{tgt_lang}: "
```

For terminology-constrained translation tasks:

```
"Translate the following text from {src_lang} into {tgt_lang} using the provided terminology pairs, ensuring the specified terms are accurately translated as indicated.\nTerminology pairs: {term_text}\n{src_lang}: {src}\n{tgt_lang}: "
```

For Automatic Post-Editing (APE) tasks:
```
"Improve the following machine-generated translation from {src_lang} to {tgt_lang}. Correct errors and generate a more accurate translation.\n{src_lang}: {src}\n{tgt_lang}: {mt_text}\n{tgt_lang}: "
```

# 🔥 Training 
Training consists of two stages: first, the Adaptor and Decoder are trained using bilingual data; second, all model parameters are fine-tuned using ComMT translation data.

Prepare your data directory as follows:

```
LaMaTE/
├── data/
│   ├── wmt23-sample10M/ # for stage1 training
│   │   ├── zh-en/
│   │   │   ├── train.zh-en.general_trans.jsonq
│   │   │   ├── valid.zh-en.general_trans.json
│   │   │   ├── test.en-zh.general_trans.wmt23.json
│   │   │   └── test.zh-en.general_trans.wmt23.json
│   │   └── de-en/
│   │       └── xxx
│   │
│   │── ComMT/ # for stage2 training
│   │   ├── zh-en/
│   │   │   ├── train.zh-en.ape.json
│   │   │   ├── train.zh-en.doc_trans.json
│   │   │   ├── train.zh-en.general_trans.json
│   │   │   └── xxx  # other translation task data
│   │   └── de-en/
│   │       └── xxx
```

Maintain a consistent file names: ```train/valid.${first_lang}-en.${task_type}.json```. 
Test sets should clearly specify the direction of translation. 

The ```task_types``` values are:
- general_trans
- doc_trans
- domain_medical,domain_law,domain_it,domain_literature,domain_colloquial
- term_con_trans
- ape
- context_learning_trans

Each line in the data files represents a sample, labeled according to the task_type key. 
For more details, refer to [ComMT](https://huggingface.co/datasets/NiuTrans/ComMT).

To train:
```
cd scripts
bash train_lamate_stage1.sh
bash train_lamate_stage2.sh
```

For training commands and configurations, please follow the provided ```scripts``` in the scripts directory.

# ⚡ Inference 
After training, perform batch inference on the ComMT test set:

```
bash inference_lamate.sh
```
Results are saved in ```${model_dir}/decoder_result```.

# 📊 Evaluation
Evaluate using BLEU and COMET:

```
bash eval_commt.sh ${decoder_result_dir}
```
Results are stored in ```scripts/ComMT_result.xlsx```.

# Reference
For more details, please refer to LaMaTE [paper](https://arxiv.org/abs/2503.06594).

Email: luoyingfeng_neu@outlook.com
```
@misc{luoyf2025lamate,
      title={Beyond Decoder-only: Large Language Models Can be Good Encoders for Machine Translation}, 
      author={Yingfeng Luo, Tong Zheng, Yongyu Mu, Bei Li, Qinghong Zhang, Yongqi Gao, Ziqiang Xu, Peinan Feng, Xiaoqian Liu, Tong Xiao, Jingbo Zhu},
      year={2025},
      eprint={2503.06594},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
