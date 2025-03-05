# !/bin/bash
set -eux
ROOT_DIR=$(dirname $(dirname `readlink -f $0`))

comet_model=/mnt/luoyingfeng/model_card/wmt22-comet-da/checkpoints/model.ckpt 

decode_dir=${1:-"/mnt/luoyingfeng/llm4nmt/exps/Meta-Llama-3-8B/sft_ComMT/decode_result"}

hypo_files=(
$decode_dir/test-cs-en-doc_trans
$decode_dir/test-cs-en-domain_medical
$decode_dir/test-de-en-ape
$decode_dir/test-de-en-doc_trans
$decode_dir/test-de-en-domain_colloquial
$decode_dir/test-de-en-domain_literature
$decode_dir/test-de-en-domain_medical
$decode_dir/test-de-en-general_trans-wmt23
$decode_dir/test-de-en-term_con_trans
$decode_dir/test-en-cs-doc_trans
$decode_dir/test-en-cs-domain_it
$decode_dir/test-en-cs-domain_law
$decode_dir/test-en-cs-general_trans-wmt23
$decode_dir/test-en-cs-term_con_trans
$decode_dir/test-en-de-ape
$decode_dir/test-en-de-doc_trans
$decode_dir/test-en-de-domain_colloquial
$decode_dir/test-en-de-domain_law
$decode_dir/test-en-de-domain_medical
$decode_dir/test-en-de-general_trans-wmt23
$decode_dir/test-en-de-term_con_trans
$decode_dir/test-en-ru-ape
$decode_dir/test-en-ru-doc_trans
$decode_dir/test-en-ru-domain_colloquial
$decode_dir/test-en-ru-domain_it
$decode_dir/test-en-ru-domain_medical
$decode_dir/test-en-ru-general_trans-wmt23
$decode_dir/test-en-ru-term_con_trans
$decode_dir/test-en-zh-ape
$decode_dir/test-en-zh-doc_trans
$decode_dir/test-en-zh-domain_colloquial
$decode_dir/test-en-zh-domain_literature
$decode_dir/test-en-zh-domain_medical
$decode_dir/test-en-zh-general_trans-wmt23
$decode_dir/test-en-zh-term_con_trans
$decode_dir/test-ru-en-ape
$decode_dir/test-ru-en-domain_colloquial
$decode_dir/test-ru-en-domain_literature
$decode_dir/test-ru-en-domain_medical
$decode_dir/test-ru-en-general_trans-wmt23
$decode_dir/test-zh-en-doc_trans
$decode_dir/test-zh-en-domain_colloquial
$decode_dir/test-zh-en-domain_literature
$decode_dir/test-zh-en-domain_medical
$decode_dir/test-zh-en-general_trans-wmt23
$decode_dir/test-zh-en-term_con_trans
)
            
src_file_strs=""
ref_file_strs=""
hypo_file_strs=""
lang_pair_strs=""

for hypo_file in ${hypo_files[@]}; do 

    filename=$(basename "$hypo_file")

    filename=${filename#test-}
    filename=${filename%-new}
    IFS='-' read -r src_lang tgt_lang task_type <<< "$filename"
    
    if [ "$src_lang" != "en" ]; then
        first_lang="$src_lang"
    else
        first_lang="$tgt_lang"
    fi
    
    lp=${src_lang}-${tgt_lang}
    lp2=${src_lang}2${tgt_lang}


    src_file=$ROOT_DIR/data/ComMT_txt/${first_lang}-en/test.$lp.$task_type.$src_lang.txt
    ref_file=$ROOT_DIR/data/ComMT_txt/${first_lang}-en/test.$lp.$task_type.$tgt_lang.txt

    src_file_strs=${src_file_strs:+$src_file_strs,}$src_file
    ref_file_strs=${ref_file_strs:+$ref_file_strs,}$ref_file
    hypo_file_strs=${hypo_file_strs:+$hypo_file_strs,}$hypo_file
    lang_pair_strs=${lang_pair_strs:+$lang_pair_strs,}$lp2
        
done


python $ROOT_DIR/mt_scoring.py \
    --metric "bleu,comet_22"  \
    --comet_22_path $comet_model \
    --lang_pair $lang_pair_strs \
    --src_file $src_file_strs \
    --ref_file $ref_file_strs \
    --hypo_file $hypo_file_strs \
    --record_file "ComMT_result.xlsx" \
    --write_key "suffix" \
    --gpu 0
