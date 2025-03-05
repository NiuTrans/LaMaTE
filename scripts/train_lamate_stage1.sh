#! /bin/bash
set -eux
ROOT_DIR=$(dirname $(dirname `readlink -f $0`))

export HF_HOME="./cache/"
export HF_DATASETS_CACHE="./cache/huggingface_cache/datasets"

## only data parallel
config_file=$ROOT_DIR/configs/accelerate_config_6gpu.yaml

## model
model_dir=/mnt/luoyingfeng/model_card/Meta-Llama-3-8B
# resume_from_checkpoint=xx
run_mode="init"

model_method="lamate"
encoder_method="stack"
encoder_layer_num=8
decoder_layer_num=8
decoder_hidden_size=1024
decoder_intermediate_size=2752
decoder_num_attention_heads=16
decoder_num_key_value_heads=16

decoder_param_method="freeze"
tag=lamate_s1

## data
language_pairs=de-en,en-de,cs-en,en-cs,ru-en,en-ru,zh-en,en-zh
mmt_data_path=$ROOT_DIR/data/wmt23-sample10M
trans_task="general_trans"
epoch=1
batch_size=40
gradient_accumulation=16

## save
output_dir=$ROOT_DIR/exps/$tag
mkdir -p $output_dir
cp $0 $output_dir


accelerate launch --config_file $config_file $ROOT_DIR/src/run_seq2seq_mt.py \
    --model_name_or_path $model_dir \
    --resume_from_checkpoint ${resume_from_checkpoint:-""} \
    --encoder_layer_num ${encoder_layer_num} \
    --decoder_layer_num $decoder_layer_num \
    --decoder_hidden_size $decoder_hidden_size \
    --decoder_intermediate_size $decoder_intermediate_size \
    --decoder_num_attention_heads $decoder_num_attention_heads \
    --decoder_num_key_value_heads $decoder_num_key_value_heads \
    --encoder_method $encoder_method \
    --model_method ${model_method:-"norm"} \
    --run_mode ${run_mode:-""} \
    --decoder_param_method ${decoder_param_method:-"share"} \
    --mmt_data_path $mmt_data_path \
    --trans_task $trans_task \
    --test_dataname wmt23 \
    --language_pairs $language_pairs \
    --use_fast_tokenizer \
    --do_eval \
    --do_train \
    --do_predict \
    --learning_rate 5e-4 \
    --weight_decay 0.01 \
    --lr_scheduler_type inverse_sqrt \
    --warmup_ratio 0.01 \
    --load_best_model_at_end  \
    --cache_dir ./cache \
    --dataloader_num_workers 8 \
    --preprocessing_num_workers 16 \
    --dataloader_prefetch_factor 64 \
    --dataloader_persistent_workers  \
    --max_source_length 256 \
    --max_target_length 256 \
    --output_dir  $output_dir \
    --num_train_epochs $epoch \
    --patience 3 \
    --per_device_train_batch_size $batch_size \
    --per_device_eval_batch_size $batch_size \
    --gradient_accumulation_steps $gradient_accumulation \
    --predict_with_generate \
    --num_beams 5 \
    --max_new_tokens 256 \
    --evaluation_strategy steps \
    --save_strategy steps \
    --logging_strategy steps \
    --eval_steps  2000 \
    --save_steps 2000 \
    --logging_steps  50 \
    --save_total_limit  5 \
    --fp16 \
    --seed 42 \
    --report_to "tensorboard" \
    --overwrite_output_dir True \
   | tee $output_dir/train.log
    