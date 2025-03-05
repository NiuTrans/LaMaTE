#! /bin/bash
set -eux
ROOT_DIR=$(dirname $(dirname `readlink -f $0`))

export HF_HOME="./cache/"
export HF_DATASETS_OFFLINE=1


model_dir=/mnt/luoyingfeng/model_card/Meta-Llama-3-8B


config_file=$ROOT_DIR/configs/deepspeed_train_config_bf16_6gpu.yaml

# resume_from_checkpoint=xxx

mmt_data_path=$ROOT_DIR/data/ComMT
language_pairs=de-en,en-de,cs-en,en-cs,ru-en,en-ru,zh-en,en-zh
trans_task="general_trans,doc_trans,domain_medical,domain_law,domain_it,domain_literature,domain_colloquial,term_con_trans,ape,context_learning_trans"
predict_task="general_trans,doc_trans,domain_medical,domain_law,domain_it,domain_literature,domain_colloquial,term_con_trans,ape"
# predict_task="general_trans"

tag=sft_ComMT

output_dir=$ROOT_DIR/exps/$tag
mkdir -p $output_dir
cp $0 $output_dir


accelerate launch --config_file $config_file --main_process_port 28500 $ROOT_DIR/src/run_llm_mt.py \
    --model_name_or_path $model_dir \
    --resume_from_checkpoint ${resume_from_checkpoint:-""} \
    --trans_task $trans_task \
    --predict_task $predict_task \
    --mmt_data_path $mmt_data_path \
    --test_dataname wmt23 \
    --cache_dir ./cache \
    --use_fast_tokenizer \
    --do_train \
    --do_eval \
    --do_predict \
    --load_best_model_at_end \
    --language_pairs $language_pairs \
    --low_cpu_mem_usage \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --lr_scheduler_type inverse_sqrt \
    --warmup_ratio 0.01 \
    --ignore_pad_token_for_loss \
    --ignore_prompt_token_for_loss \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy steps \
    --save_strategy steps \
    --logging_strategy steps \
    --eval_steps 0.1  \
    --save_steps 0.1 \
    --logging_steps 0.01 \
    --save_total_limit 5 \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 8 \
    --max_source_length 512 \
    --max_new_tokens 512 \
    --output_dir  $output_dir \
    --overwrite_output_dir True \
    --num_beams 5 \
    --predict_with_generate \
    --fp16 \
    --seed 42 \
    --report_to "tensorboard" \
    --overwrite_cache False | tee $output_dir/train.log


