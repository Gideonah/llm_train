# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import copy
import datasets


def get_preprocessed_sapher(dataset_config, tokenizer, split):
    dataset = datasets.load_dataset("Gideonah/sapher_test", split=split)
   
    def tokenize_add_label(sample):
        prompt = tokenizer.encode(sample["text"], add_special_tokens=False)
        
        sample = {
            "input_ids": prompt,
            "attention_mask" : [1] * len(prompt),
            "labels": prompt,
            }

        return sample

    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

    return dataset
