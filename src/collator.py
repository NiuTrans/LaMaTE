import random
from collections.abc import Mapping
import numpy as np
import torch
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from transformers.utils import PaddingStrategy


@dataclass
class DataCollatorForLamate:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        pad_token_id = self.tokenizer.pad_token_id
        input_ids = [feature["input_ids"] for feature in features]
        max_length = max(len(l) for l in input_ids)
        max_length = (
                    (max_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
        )
        # left padding
        input_ids = [[pad_token_id]*(max_length-len(ids)) + ids for ids in input_ids]
        attention_mask = [[0 if x == pad_token_id else 1 for x in y] for y in input_ids]

        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None

        ## for predict only
        if labels is None:
            features = {
                "input_ids": torch.tensor(np.array(input_ids).astype(np.int64)),
                "attention_mask":  torch.tensor(np.array(attention_mask).astype(np.int64)),
            }
            return features

        ## add eos to the end of labels 
        if labels[0][-1] != self.tokenizer.eos_token_id:
            labels = [label + [self.tokenizer.eos_token_id] for label in labels]

        ## add bos to the end of labels 
        if labels[0][0] != self.tokenizer.bos_token_id:
            labels = [[self.tokenizer.bos_token_id] + label  for label in labels]

        decoder_input_ids = [label[:-1] for label in labels]
        labels = [label[1:] for label in labels]
        
        ## padding decoder_input_ids with right side 
        max_length = max(len(l) for l in decoder_input_ids)
        max_length = (
                    (max_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
        )
        decoder_input_ids = [ids + [pad_token_id]*(max_length-len(ids)) for ids in decoder_input_ids]
        decoder_attention_mask = [[0 if x == pad_token_id else 1 for x in y] for y in decoder_input_ids]
        
        ## construct labels
        labels = [label + [pad_token_id]*(max_length-len(label)) for label in labels]
        labels = [[self.label_pad_token_id if x == pad_token_id else x for x in y] for y in labels]

        features = {
            "input_ids": torch.tensor(np.array(input_ids).astype(np.int64)),
            "attention_mask":  torch.tensor(np.array(attention_mask).astype(np.int64)),
            "decoder_input_ids":  torch.tensor(np.array(decoder_input_ids).astype(np.int64)),
            "decoder_attention_mask":  torch.tensor(np.array(decoder_attention_mask).astype(np.int64)),
            "labels":  torch.tensor(np.array(labels).astype(np.int64)),
        }
        return features

    
@dataclass
class DataCollatorForCausalLM:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        input_ids = [feature["input_ids"] for feature in features]
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.

        ## padding input_ids and labels with left side and get attention_mask
        max_length = max(len(l) for l in input_ids)
        if self.pad_to_multiple_of:
            max_length = (
                        (max_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
            )
        input_ids = [[self.tokenizer.pad_token_id]*(max_length-len(ids)) + ids for ids in input_ids]
        labels = [[-100]*(max_length-len(ids)) + ids for ids in labels] if labels is not None else None
        attention_mask = [[0 if x == self.tokenizer.pad_token_id else 1 for x in y] for y in input_ids]

        features = {
            "input_ids": torch.tensor(np.array(input_ids).astype(np.int64)),
            "attention_mask":  torch.tensor(np.array(attention_mask).astype(np.int64)),
        }
        if labels is not None:
            features["labels"] = torch.tensor(np.array(labels).astype(np.int64)) 

        # print(input_ids)
        # print(attention_mask)
        # print(labels)

        return features
    