import torch
import numpy as np

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

    def __post_init__(self):
        self.return_tensors = 'pt'

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        batch = self.generate_labels(batch)
        return batch

    def generate_labels(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare binary labels for mortality prediction. Dynamically select a visit and remove all inputs after that visit.
        Based on the number of remaining days in the patient's life and the threshold, set the label.
        """
        new_batch = {}
        for input_ids, token_type_ids, attention_mask, eol_days in zip(*batch.values()):
            num_visits = np.max(token_type_ids.numpy())
            if num_visits == 1:
                cutoff_visit = 1
            else:
                cutoff_visit = rng.integers(1, num_visits, endpoint=True)
            last_visit_ind = np.max(np.nonzero(token_type_ids.numpy() == cutoff_visit))
            cutoff_ind = last_visit_ind + 1
            label = int(eol_days[last_visit_ind] < self.eol_threshold)

            # Remove items after cutoff
            input_ids[cutoff_ind:] = self.tokenizer.pad_token_id
            input_ids[cutoff_ind] = self.tokenizer.sep_token_id  # Put a [SEP] token at the cutoff point

            token_type_ids[cutoff_ind:] = 0  # [SEP] and [PAD] have token_type_id of 0
            attention_mask[cutoff_ind + 1:] = 0  # After [SEP], we mask attention

            new_batch.setdefault("input_ids", []).append(input_ids.tolist())
            new_batch.setdefault("token_type_ids", []).append(token_type_ids.tolist())
            new_batch.setdefault("attention_mask", []).append(attention_mask.tolist())
            new_batch.setdefault("labels", []).append(label)
        
        new_batch = {k: torch.as_tensor(np.array(v)) for k, v in new_batch.items()}
        return new_batch  

@dataclass
class DataCollatorForDiseasePrediction(DataCollatorMixin):
    """
    Data collator. Accepts batches of ids for patients (input_ids, token_type_ids, attention_mask). Selects random visit number, masks tokens after (and including) that visit. 
    Produces ICD-10 label. Derivative of "DataCollatorForNextDiseasePrediction".

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
    """
    
    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __post_init__(self):
        self.return_tensors = 'pt'

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        batch = self.batch_data_collator(batch)
        return batch

    def torch_mask_tokens(self, inputs: Any, input_types: Any, attention_mask: Any) -> Tuple[Any, Any, Any, Any]:
        """
        Prepare masked tokens inputs/labels for next disease prediction: MASK all of last visit. Set labelss to only have first disease code of last visit
        """
        import torch

        # get the first disease from the last visit and set labels to only include that one token
        next_disease_indices = torch.argmax(input_types, dim=1)
        labels = torch.stack([t[i:i+1] for i, t in zip(next_disease_indices, inputs)])

        # turn the next visit into just a mask token then a sep token
        post_next_disease_indices = torch.stack([torch.arange(0, inputs.shape[1]) >= i for i in next_disease_indices])
        inputs[post_next_disease_indices] = self.tokenizer.pad_token_id
        input_types[post_next_disease_indices] = self.tokenizer.pad_token_type_id
        attention_mask[post_next_disease_indices] = 0
        for i, j in enumerate(next_disease_indices):
            inputs[i, j] = self.tokenizer.sep_token_id
            attention_mask[i, j] = 1

        return inputs, input_types, attention_mask, labels
    
    def lookup_ICD_10_chapter(self, code):
        """
        Accepts ICD-9 or ICD-10 codes, returns ICD-10 chapter.
        """
        if code[-2:] == '-9':
            if code[0].isalpha():
                chapter = '20' 
                return chapter
            code = code[:-2]    # 2765-9 -> 2765
            if len(code) == 5:
                code = str(code)[:-2] + "." + str(code)[-2:]    # 27939 -> 279.39  
            if len(code) == 4:
                code = str(code)[:-1] + "." + str(code)[-1]     # 0032 -> 003.2  
            while code[0] == "0":  
                code = code[1:]  # 0032 -> 3.2
            code = float(code)
            code_dict = {'1': [1,139], '2': [140,239], '3': [240,279], '4': [280,289], '5': [290,319], '6': [320,359],'7': [360,379],
                        '8': [380,389],'9': [390,459], '10': [460,519], '11': [520,579], '12': [580,629], '13': [630,679], '14': [680,709],
                        '15': [710,739], '16': [740,759], '17': [760,779], '18': [780,799], '19': [800,999]}                                                                                                  
            for i in range(1, len(code_dict) + 1):
                if code_dict[str(i)][0] <= np.floor(code) <= code_dict[str(i)][1]:
                    chapter = str(i)
                    return chapter

        if code[-3:] == '-10':
            if code[:3] == 'O9A':
                chapter = '19'
                return chapter
            code = code[:3]
            code_dict = {'1': ['A00', 'B99'], '2': ['C00', 'D49'], '3': ['D50', 'D99'], '4': ['E00', 'E99'], '5': ['F00', 'F99'], '6': ['G00', 'G99'],'7': ['H00', 'H59'],
                                '8': ['H60', 'H99'],'9': ['I00', 'I99'], '10': ['J00', 'J99'], '11': ['K00', 'K99'], '12': ['L00', 'L99'], '13': ['M00', 'M99'], '14': ['N00', 'N99'], 
                                '15': ['O00', 'O99'], '16': ['P00', 'P99'], '17': ['Q00', 'Q99'], '18': ['R00', 'R99'], '19': ['S00', 'T99'], '20': ['V00', 'Y99'], '21': ['Z00', 'Z99'], 
                                '22': ['U00', 'U99']}  
            for i in range(1, len(code_dict)):
                if code_dict[str(i)][0] <= code <= code_dict[str(i)][1]:
                    chapter = str(i)
                    return chapter   

    def data_collator(self, batch, i):
        current = {'input_ids':batch['input_ids'][i], 'token_type_ids':batch['token_type_ids'][i], 'attention_mask':batch['attention_mask'][i]}
        max_token = max(current['token_type_ids'].tolist())
        if max_token == 1:
            batch['labels'].append(-100)
            return batch
        else:
            visit_to_not_keep = np.random.randint(2,max_token+1) 
            first_index_to_not_keep = current['token_type_ids'].tolist().index(visit_to_not_keep)
            label = self.tokenizer.convert_ids_to_tokens(current['input_ids'][first_index_to_not_keep].item())
            label = self.lookup_ICD_10_chapter(label)
            batch['labels'].append(int(label))
            current['input_ids'][first_index_to_not_keep] = self.tokenizer.convert_tokens_to_ids('[SEP]')
            current['input_ids'][first_index_to_not_keep+1:] = self.tokenizer.convert_tokens_to_ids('[PAD]')
            current['token_type_ids'][first_index_to_not_keep:] = 0
            current['attention_mask'][first_index_to_not_keep:] = 0
            batch['input_ids'][i] = current['input_ids']
            batch['token_type_ids'][i] = current['token_type_ids']
            batch['attention_mask'][i] = current['attention_mask']
            return batch   

    def batch_data_collator(self, batch):
        batch_size = len(batch['input_ids'])
        batch['labels'] = []
        for i in range(batch_size):
            batch = self.data_collator(batch, i)
        batch['labels'] = torch.LongTensor(batch['labels'])
        return batch 
