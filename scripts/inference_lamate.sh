#! /bin/bash
set -eux
ROOT_DIR=$(dirname $(dirname `readlink -f $0`))

export HF_HOME="./cache"
export HF_DATASETS_OFFLINE=1

config_file=$ROOT_DIR/configs/accelerate_config_6gpu.yaml

language_pairs=de-en,en-de,cs-en,en-cs,ru-en,en-ru,zh-en,en-zh

mmt_data_path=$ROOT_DIR/data/ComMT
trans_task="general_trans,doc_trans,domain_medical,domain_law,domain_it,domain_literature,domain_colloquial,term_con_trans,ape,context_learning_trans"
predict_task="general_trans,doc_trans,domain_medical,domain_law,domain_it,domain_literature,domain_colloquial,term_con_trans,ape"
# predict_task="general_trans"


model_dir=xxxx
model_method="lamate"

output_dir=$model_dir
mkdir -p $output_dir
cp $0 $output_dir

accelerate launch --config_file $config_file --main_process_port 26000 $ROOT_DIR/src/run_seq2seq_mt.py \
    --model_name_or_path $model_dir \
    --model_method $model_method \
    --mmt_data_path $mmt_data_path \
    --trans_task $trans_task \
    --predict_task $predict_task \
    --test_dataname wmt23 \
    --language_pairs $language_pairs \
    --use_fast_tokenizer \
    --do_predict \
    --predict_with_generate \
    --num_beams 5 \
    --max_new_tokens 512 \
    --cache_dir ./cache \
    --dataloader_num_workers 4 \
    --max_source_length 512 \
    --max_target_length 512 \
    --output_dir  $output_dir \
    --per_device_eval_batch_size 8 \
    --fp16 \
    --seed 42 
