import torch
import numpy as np
import itertools

from transformers.data.data_collator import DataCollatorMixin, _torch_collate_batch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Tuple

rng = np.random.default_rng()

@dataclass
class DataCollatorForMortalityPrediction(DataCollatorMixin):
    """
    Data collator used for mortality. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        eol_threshold (`int`, *optional*):
            Threshold on the number of days left in a patient's life to consider for mortality prediction.
            E.g. `eol_threshold=30`: patients that died within a month of the visit are considered label 1,
            the remainder are label 0. If None, patients with any non-nan days left in life are labelled 1.
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
    """

    tokenizer: PreTrainedTokenizerBase
    eol_threshold: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    inference_metadata: bool = False

    def __post_init__(self):
        self.return_tensors = 'pt'

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = {}
        for (
            patient_id, visit_id, visit_date, days_until_deceased,
            input_ids, token_type_ids, attention_mask, special_tokens_mask
        ) in map(lambda x: x.values(), examples):
            num_visits = np.max(np.array(token_type_ids))
            if num_visits == 1:
                cutoff_visit = 1
            else:
                cutoff_visit = rng.integers(1, num_visits, endpoint=True)
            last_visit_ind = np.max(np.nonzero(np.array(token_type_ids) == cutoff_visit))
            
            # Select data from last visit
            label = int(days_until_deceased[last_visit_ind] < self.eol_threshold)
            batch.setdefault("labels", []).append(label)
            if self.inference_metadata:
                batch.setdefault("patient_id", []).append(patient_id[last_visit_ind])
                batch.setdefault("visit_id", []).append(visit_id[last_visit_ind])
                batch.setdefault("visit_date", []).append(visit_date[last_visit_ind])
            
            # Remove items after cutoff
            cutoff_ind = last_visit_ind + 1

            input_ids = np.array(input_ids)
            input_ids[cutoff_ind:] = self.tokenizer.pad_token_id
            input_ids[cutoff_ind] = self.tokenizer.sep_token_id  # Put a [SEP] token at the cutoff point

            token_type_ids = np.array(token_type_ids)
            token_type_ids[cutoff_ind:] = 0  # [SEP] and [PAD] have token_type_id of 0
            attention_mask = np.array(attention_mask)
            attention_mask[cutoff_ind + 1:] = 0  # After [SEP], we mask attention
            
            batch.setdefault("input_ids", []).append(input_ids)
            batch.setdefault("token_type_ids", []).append(token_type_ids)
            batch.setdefault("attention_mask", []).append(attention_mask)
        
        # Convert relevant items to torch tensors
        for key in ["input_ids", "token_type_ids", "attention_mask", "labels"]:
            batch[key] = torch.as_tensor(np.array(batch[key]))
        return batch