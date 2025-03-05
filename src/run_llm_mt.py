# coding=utf-8

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

import datasets
import torch

import transformers
from transformers import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
    EarlyStoppingCallback
)

from transformers.utils.versions import require_version

from trainer import LlmmtTrainer
from utils import (
    load_mmt_dataset, 
    process_mmt_data_for_llm, 
    clean_outputstring, 
    load_tokenizer, 
    load_model, 
    LANG_TABLE
)
from collator import  DataCollatorForCausalLM

from transformers import MODEL_FOR_CAUSAL_LM_MAPPING

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    do_sample: bool = field(
        default=False,
    )

    ## new
    architecture: Optional[str] = field(default="decoder-only")

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    language_pairs: str = field(default="", metadata={"help": "training language pairs"})
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    mmt_data_path: Optional[str] = field(default=None, metadata={"help": "The input MMT training data path."})
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes, truncate the number of test examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    override_test_data_path: Optional[str] = field(default=None, metadata={"help": "This will override the default test data in the mmt data"})
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    ignore_prompt_token_for_loss: bool = field(
        default=False,
        metadata={
            "help": "Whether to ignore the prompt tokens in the loss computation or not."
        },
    )
    max_source_length: Optional[int] = field(
        default=256,
        metadata={
            "help": (
                "The maximum total sequence length text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_new_tokens: Optional[int] = field(
        default=256,
        metadata={
            "help": (
                "The maximum new tokens to generate except the prompt."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=5,
        metadata={
            "help": (
                "Beam size for generation"
            )
        }
    )

    display_num_translations: Optional[int] = field(
        default=10,
        metadata={
            "help": (
                "Number of translations will be displayed after translation."
            )
        }
    )

    right_pad: bool = field(
        default=False,
        metadata={
            "help": "Use right pad for training, especially for models like MPT."
        },
    )

    trans_task: str = field(
        default="general_trans",
        metadata={
            "help": "train task"
        },
    )
    predict_task: str = field(
        default="general_trans",
        metadata={
            "help": "train task"
        },
    )
    test_dataname: str = field(
        default="wmt23",
        metadata={
            "help": "Use for general_trans, support wmt23, wmt22, flores"
        },
    )

    suffix: Optional[str] = field(default="", metadata={"help": "The suffix of the training file."})

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # load and set special token id of tokenizer
    tokenizer = load_tokenizer(data_args, model_args, training_args, logger)
    # Load model
    model = load_model(data_args, model_args, training_args, tokenizer, logger)

    # Get the datasets
    pairs = set(data_args.language_pairs.split(","))
    trans_task = data_args.trans_task.split(",")
    logger.info(f"Training lanauage pairs: {pairs}\nTraining translation task: {trans_task}")

    train_raw_data, valid_raw_data, test_raw_data = None, None, None
    if data_args.mmt_data_path:
        train_raw_data, valid_raw_data, test_raw_data = load_mmt_dataset(pairs, trans_task, data_args, model_args, training_args, logger)
        train_datasets, eval_datasets, test_datasets = process_mmt_data_for_llm(train_raw_data, valid_raw_data, test_raw_data, pairs, tokenizer, data_args, training_args)

    # load tokenizer
    set_seed(training_args.seed)
    
    # collate_fn = default_data_collator
    collate_fn = DataCollatorForCausalLM(tokenizer, model=model, pad_to_multiple_of=8 if training_args.fp16 else None )
    
    # Initialize our Trainer
    trainer = LlmmtTrainer(
        model=model,
        args=training_args,
        train_dataset=train_datasets if training_args.do_train else None,
        eval_dataset=eval_datasets if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=collate_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint:
            checkpoint = training_args.resume_from_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        trainer.save_state()
        trainer.save_model()  # Saves the tokenizer too for easy upload
    
    # Prediction
    predict_tasks = data_args.predict_task.split(",")
    if training_args.do_predict:
        trainer.args.prediction_loss_only = False
        lg_pairs = sorted(test_datasets.keys()) # make sure each device print in the same order
        for lg_pair in lg_pairs:
            cur_test_dataset = test_datasets[lg_pair]
            src_lang, tgt_lang = lg_pair.split("-")
            for task in cur_test_dataset.keys():
                if task not in predict_tasks:
                    logger.info(f"skip predict {lg_pair}.{task}")
                    continue
                task_test_dataset = cur_test_dataset[task]
                logger.info(f"*** Prediction for {lg_pair}.{task} ***")
                preds, _, _ = trainer.predict(
                    test_dataset=task_test_dataset, 
                    max_new_tokens=data_args.max_new_tokens, 
                    num_beams=data_args.num_beams, 
                    metric_key_prefix="test",
                    use_cache=True,
                    do_sample=model_args.do_sample
                )

                # Replace -100s used for padding as we can't decode them
                if int(torch.cuda.current_device()) == 0:
                    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
                    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

                    # Some simple post-processing
                    decoded_preds = [pred.strip() for pred in decoded_preds]

                    for idx in range(data_args.display_num_translations):
                        print("------------------------")
                        print(decoded_preds[idx])
                    decode_dir = os.path.join(training_args.output_dir, "decode_result")
                    os.makedirs(decode_dir, exist_ok=True)
                    predic_file_name = f"test-{src_lang}-{tgt_lang}-{task}"
                    if task == "general_trans":
                        predic_file_name += f"-{data_args.test_dataname}"
                    output_prediction_file = os.path.join(decode_dir, predic_file_name)
                    with open(output_prediction_file, "w", encoding="utf-8") as f:
                        if task != "ape":
                            suffix = f"\n{LANG_TABLE[tgt_lang]}:"
                        else:
                            suffix = "Improved translation: "
                        for pred in decoded_preds:
                            pred = clean_outputstring(pred, suffix, logger, split_idx=1)
                            f.writelines([pred, "\n"])

if __name__ == "__main__":
    main()
