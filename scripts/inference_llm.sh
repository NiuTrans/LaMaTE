#! /bin/bash
set -eux
ROOT_DIR=$(dirname $(dirname `readlink -f $0`))

export HF_HOME="./cache/"
export HF_DATASETS_OFFLINE=1

config_file=$ROOT_DIR/configs/accelerate_config_6gpu.yaml

language_pairs=de-en,en-de,cs-en,en-cs,ru-en,en-ru,zh-en,en-zh
trans_task="general_trans,doc_trans,domain_medical,domain_law,domain_it,domain_literature,domain_colloquial,term_con_trans,ape,context_learning_trans"
predict_task="general_trans,doc_trans,domain_medical,domain_law,domain_it,domain_literature,domain_colloquial,term_con_trans,ape"
test_dataname=wmt23

mmt_data_path=$ROOT_DIR/data/ComMT

model_dir=xxx

output_dir=$model_dir
mkdir -p $output_dir
cp $0 $output_dir

accelerate launch --config_file $config_file --main_process_port 28500 $ROOT_DIR/src/run_llm_mt.py \
    --model_name_or_path $model_dir \
    --mmt_data_path $mmt_data_path \
    --use_fast_tokenizer \
    --do_predict \
    --predict_with_generate \
    --language_pairs $language_pairs \
    --trans_task $trans_task \
    --predict_task $predict_task \
    --test_dataname $test_dataname \
    --low_cpu_mem_usage \
    --per_device_eval_batch_size 4 \
    --output_dir  $output_dir \
    --max_source_length 512 \
    --seed 42 \
    --num_beams 5 \
    --max_new_tokens 512 \
    --overwrite_cache True \
    --torch_dtype "auto"
