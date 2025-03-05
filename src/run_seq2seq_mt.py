#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional
import torch
import time

import datasets
import numpy as np
import copy
import collator
import utils

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    M2M100Tokenizer,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
    EarlyStoppingCallback 
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from modeling_llama_seq2seq import LlamaCrossAttentionEncDec

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.38.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/translation/requirements.txt")

logger = logging.getLogger(__name__)

# A list of all multilingual tokenizer which require src_lang and tgt_lang attributes.
MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast, M2M100Tokenizer]


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )


    ## new add
    model_method: str = field(
        default="default", 
        metadata={
            "help": "The default refers to the general seq2seq model, such as t5,bart"
        },
    )
    decoder_layer_num: int = field(default=8)
    run_mode: str = field(default="resume")
    do_sample: bool = field(default=False)
    patience: int = field(default=3)
    encoder_method: str = field(default="causal")

    decoder_param_method: str = field(default="freeze")
    decoder_hidden_size: int = field(default=1024)
    decoder_intermediate_size: int = field(default=2752)
    decoder_num_attention_heads: int = field(default=16)
    decoder_num_key_value_heads: int = field(default=16)
    decoder_model_name_or_path: str = field(default=None)
    encoder_layer_num: int = field(default=8)

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=256,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=256,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`. "
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    override_test_data_path: Optional[str] = field(default=None, metadata={"help": "This will override the default test data in the mmt data"})
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
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
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
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the :obj:`decoder_start_token_id`.Useful for"
                " multilingual models like :doc:`mBART <../model_doc/mbart>` where the first generated token needs to"
                " be the target language token.(Usually it is the target language token)"
            )
        },
    )
    right_pad: bool = field(
        default=False,
        metadata={
            "help": "Use right pad for training, especially for models like MPT."
        },
    )

    ### add new
    language_pairs: str = field(default="de-en,en-de,cs-en,en-cs,ru-en,en-ru,zh-en,en-zh", metadata={"help": "training language pairs"})
    mmt_data_path: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
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

    trans_task: str = field(
        default="general_trans"
    )
    predict_task: str = field(
        default="general_trans"
    )
    test_dataname: str = field(
        default="wmt23"
    )

    def __post_init__(self):
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


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

    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.",
            FutureWarning,
        )
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry("run_translation", model_args, data_args)

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
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    pairs = set(data_args.language_pairs.split(","))
    trans_task = data_args.trans_task.split(",")
    logger.info(f"Training lanauage pairs: {pairs}\nTraining translation task: {trans_task}")
    
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code
    )

    add_eos_token = True if model_args.model_method == "default" else False
    tokenizer = utils.load_tokenizer(data_args, model_args, training_args, logger, add_eos_token=add_eos_token)
    
    # t5
    if model_args.model_method == "default":
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
    
    elif model_args.model_method == "lamate":
        # stage 1
        if model_args.run_mode == "init":
            # seting decoder config
            decoder_config = copy.deepcopy(config.to_dict())
            decoder_config["num_hidden_layers"] = model_args.decoder_layer_num
            decoder_config["num_encoder_layers"] = config.num_hidden_layers
            decoder_config["decoder_param_method"] = model_args.decoder_param_method
            decoder_config["model_method"] = model_args.model_method
            decoder_config["hidden_size"] = model_args.decoder_hidden_size
            decoder_config["intermediate_size"] = model_args.decoder_intermediate_size
            decoder_config["num_attention_heads"] = model_args.decoder_num_attention_heads
            decoder_config["num_key_value_heads"] = model_args.decoder_num_key_value_heads
            config.decoder =  decoder_config
            # set encoder config
            config.use_cache = False
            config.is_encoder_decoder = True
            config.decoder_start_token_id = config.bos_token_id
            config.encoder_method = model_args.encoder_method
            config.encoder_layer_num = model_args.encoder_layer_num
            # make param dict
            state_dict = utils.make_model_state_dict(model_path=model_args.model_name_or_path)
            model = LlamaCrossAttentionEncDec.from_pretrained(None, config=config, state_dict=state_dict, ignore_mismatched_sizes=True)
            model.freeze_llm() # frozen LLM
        # stage 2
        else:
            model = LlamaCrossAttentionEncDec.from_pretrained(model_args.model_name_or_path, config=config)
    else:
        print("Not implement this model yet!")
        exit()

    model = utils.set_model_special_tokens(model, model_args.model_name_or_path)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if data_args.mmt_data_path is not None:
        train_raw_data, valid_raw_data, test_raw_data = utils.load_mmt_dataset(pairs, trans_task, data_args, model_args, training_args, logger)
        train_datasets, eval_datasets, test_datasets = utils.process_mmt_data_for_seq2seq(train_raw_data, valid_raw_data, test_raw_data, pairs, tokenizer, data_args, training_args)

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for "
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        if model_args.model_method == "default":
            data_collator = DataCollatorForSeq2Seq(
                tokenizer,
                model=model,
                label_pad_token_id=label_pad_token_id,
                pad_to_multiple_of=8 if training_args.fp16 else None,
            )
        elif model_args.model_method in ["lamate"]:
            data_collator = collator.DataCollatorForLamate(
                tokenizer,
                model=model,
                label_pad_token_id=label_pad_token_id,
                pad_to_multiple_of=8 if training_args.fp16 else None,
            )
        else:
            print("Not implement this model yet!")
            exit()


    optimizer = None

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_datasets if training_args.do_train else None,
        eval_dataset=eval_datasets if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=model_args.patience)],
        optimizers=(optimizer, None)
    )

    logger.info(model)
    if training_args.do_train:
        utils.print_trainable_parameters(model)

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_datasets)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_datasets))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    
    predict_tasks = data_args.predict_task.split(",")
    if training_args.do_predict:
        lg_pairs = sorted(test_datasets.keys())
        for lg_pair in lg_pairs:
            cur_test_dataset = test_datasets[lg_pair]
            src_lang, tgt_lang = lg_pair.split("-")
            for task in cur_test_dataset.keys():
                if task not in predict_tasks:
                    logger.info(f"skip predict {lg_pair}.{task}")
                    continue
                task_test_dataset = cur_test_dataset[task]
                start = time.time()
                logger.info(f"*** Prediction for {lg_pair}.{task} ***")

                predict_results = trainer.predict(
                    task_test_dataset, 
                    metric_key_prefix="test", 
                    num_beams=num_beams, 
                    max_new_tokens=data_args.max_new_tokens,
                    do_sample=model_args.do_sample
                )
                metrics = predict_results.metrics

                if int(torch.cuda.current_device()) == 0:
                    predictions = predict_results.predictions
                    if len(predictions) != len(task_test_dataset):
                        predictions = predictions[:len(task_test_dataset)]
                    num_tokens = sum([ len(t) for t in predictions ])
                    timediff = time.time() - start
                    logger.info(f"tokens/sec: {num_tokens//timediff}, time elapsed: {timediff}, num_tokens {num_tokens}")
                    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
                    predictions = tokenizer.batch_decode(
                        predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                    )
                    predictions = [pred.replace("\n", "") for pred in predictions]
                    
                    decode_dir = os.path.join(training_args.output_dir, "decode_result")
                    os.makedirs(decode_dir, exist_ok=True)
                    predic_file_name = f"test-{src_lang}-{tgt_lang}-{task}"
                    if task == "general_trans":
                        predic_file_name += f"-{data_args.test_dataname}"
                    output_prediction_file = os.path.join(decode_dir, predic_file_name)
                    with open(output_prediction_file, "w", encoding="utf-8") as writer:
                        writer.write("\n".join(predictions))

if __name__ == "__main__":
    main()
