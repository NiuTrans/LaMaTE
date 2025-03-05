#! /bin/bash
set -eux
ROOT_DIR=$(dirname $(dirname `readlink -f $0`))

export HF_HOME="./cache/"
export HF_DATASETS_OFFLINE=1

model_name=Meta-Llama-3-8B

config_file=$ROOT_DIR/configs/deepspeed_train_config_bf16_6gpu.yaml

# model_dir=/mnt/luoyingfeng/model_card/$model_name
model_dir=$ROOT_DIR/exps/Meta-Llama-3-8B/lamate_s1/checkpoint-52000
# resume_from_checkpoint=xxx
run_mode="continue"

model_method="lamate"
encoder_method="stack"
encoder_layer_num=8
decoder_layer_num=8
decoder_hidden_size=1024
decoder_intermediate_size=2752
decoder_num_attention_heads=16
decoder_num_key_value_heads=16

tag=lamate_s2

language_pairs=de-en,en-de,cs-en,en-cs,ru-en,en-ru,zh-en,en-zh
mmt_data_path=$ROOT_DIR/data/ComMT
trans_task="general_trans,doc_trans,domain_medical,domain_law,domain_it,domain_literature,domain_colloquial,term_con_trans,ape,context_learning_trans"
predict_task="general_trans,doc_trans,domain_medical,domain_law,domain_it,domain_literature,domain_colloquial,term_con_trans,ape"
# predict_task="general_trans"

output_dir=$ROOT_DIR/exps/$model_name/$tag
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
    --model_method ${model_method:-"norm"} \
    --encoder_method ${encoder_method} \
    --run_mode ${run_mode:-""} \
    --mmt_data_path $mmt_data_path \
    --trans_task $trans_task \
    --predict_task $predict_task \
    --test_dataname wmt23 \
    --language_pairs $language_pairs \
    --use_fast_tokenizer \
    --do_train \
    --do_eval \
    --do_predict \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.01 \
    --load_best_model_at_end  \
    --cache_dir ./cache \
    --dataloader_num_workers 8 \
    --preprocessing_num_workers 8 \
    --max_source_length 512 \
    --max_target_length 512 \
    --output_dir  $output_dir \
    --num_train_epochs 1 \
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 12 \
    --predict_with_generate \
    --num_beams 5 \
    --max_new_tokens 512 \
    --evaluation_strategy steps \
    --save_strategy steps \
    --logging_strategy steps \
    --eval_steps  0.1 \
    --save_steps 0.1 \
    --logging_steps  0.01 \
    --save_total_limit 5 \
    --fp16 \
    --seed 42 \
    --report_to "tensorboard" \
    --overwrite_output_dir True \
   | tee $output_dir/train.log
    