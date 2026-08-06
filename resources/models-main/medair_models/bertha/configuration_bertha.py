# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" BERTHA model configuration"""
from collections import OrderedDict
from typing import Mapping

from transformers.configuration_utils import PretrainedConfig
from transformers.onnx import OnnxConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

BERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {

}


class BerthaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BerthaModel`]. It is used to
    instantiate a BERTHA model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the original BERT
    [bert-base-uncased](https://huggingface.co/bert-base-uncased) architecture. This model is based on the original
    BERT architecture except it has configurations to add rotary position embeddings, change the position of the
    LayerNorm operation, and to add pre-configured token embeddings instead of initializing with random ones.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the BERTHA model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`BerthaModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`BerthaModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`, `"rotary"`
            For positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
            [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
            For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
            with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
            For more information on `"rotary"`, please refer to [RoFormer: Enhanced Transformer with Rotary Position
            Embedding (Su et al.)](https://arxiv.org/pdf/2104.09864.pdf).
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.
        rotary_value (`bool`, *optional*, defaults to `False`):
            Whether or not to apply rotary position embedding to the values during the attention operation. Only
            relevant if `config.position_embedding_type="rotary"`.
        layer_norm_position (`str`, *optional*, defaults to `"post"`):
            Position of the LayerNorm operation. Choose one of `"post"` or `"pre"`. For original BERT configuration,
            use `"post"`. `"post"` places the LayerNorm operations after the residual addition. `"pre"` places the
            LayerNorm operations before the attention and feed-forward network sub-blocks, outside the residual path.
            Please refer to [On Layer Normalization in the Transformer Architecture (Xiong et al.)]
            (https://arxiv.org/pdf/2002.04745.pdf) for an analysis of these different architectures.
        freeze_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether or not to freeze the word embeddings during training.
        normalize_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether or not to normalize the word embeddings (concept vectors) such that their L2 norm is 1.
        apply_ffn_to_embeddings (`str`, *optional*, defaults to `None`):
            Strategy for applying a feed-forward layer directly after the embeddings layer. Choose one of `"residual"`,
            `"plain"`, or `None`. For original BERT configuration, use `None`. `"residual"` uses a residual connection
            over the feed-forward network like in the rest of the BERTHA architecture. `"plain"` applies the standard
            feed-foward network without the residual connection. The feed-foward network is the same as the rest of the
            BERTHA architecture, including number of hidden units, activation function, and LayerNorm position.
        group_option (`str`, *optional*, defaults to `"ignore"`):
            VSA configuration for semantic pointer groups in SNOMED. Choose one of `"ignore"`, `"group_zero"`, `"isA"`,
            or `"group_vectors"`. See vsa_utils.VsaDataParser for more details.
        dp_composition (`str`, *optional*, defaults to `"snomed_all"`):
            VSA configuration for the vocabulary used to generate description pointers. Choose one of `"icd_name"`,
            `"snomed_name"`, or `"snomed_all"`. See vsa_utils.VsaDataParser for more details.
        dp_type (`str`, *optional*, defaults to `"words"`):
            VSA configuration for the composition formula to generate description pointers. Choose one of `"atomic"`,
            `"words"`, or `"words_rv"`. See vsa_utils.VsaDataParser for more details.
        path_to_vsa_parser_data (`str`, *optional*, defaults to `None`):
            Path to VSA data pickle file used to initialize the VSA parser. Only relevant if `use_vsa` is `True`.
        tokenizer_vocab (`dict`, *optional*, defaults to `None`):
            Tokenizer vocab from the tokenizer used on the data. Maps tokens to token ids. Must have the same keys as the keys
            in the VSA parser data dict.
        use_vsa (`bool`, *optional*, defaults to `False`):
            Whether or not to use the VSA configuration specified in `group_option`, `dp_composition`, and `dp_type`
            to generate the word embeddings (concept vectors). Otherwise, word embeddings will be generated from a
            random normal distribution with a standard deviation of `initializer_range`.
        path_to_pretrained_embeddings (`str`, *optional*, defaults to `None`):
            Path to a pre-generated embedding .npy file to load for the word embeddings. `use_vsa` must be set to `False`.
            Embedding dimension size must match `vocab_size` and `embedding_dim`. `freeze_word_embeddings` still applies.
        path_to_pretrained_atomics (`str`, *optional*, defaults to `None`):
            Path to a pre-generated atomic embedding .pt file to load for the atomic embeddings. `use_vsa` must be set to `True`.
            Embedding dimension size must match `parser.num_avs` and `embedding_dim`. `freeze_word_embeddings` still applies.
        use_cls (`str`, *optional*, defaults to `None`):
            Whether to use learned cls tokens with the VSA options. One of `[None, "cat", "add"]`

    Examples:

    ```python
    >>> from transformers import BerthaModel, BerthaConfig

    >>> # Initializing a BERT bert-base-uncased style configuration
    >>> configuration = BerthaConfig()

    >>> # Initializing a model from the bert-base-uncased style configuration
    >>> model = BerthaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "bertha"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        rotary_value=False,
        layer_norm_position='post',
        freeze_word_embeddings=False,
        normalize_word_embeddings=False,
        apply_ffn_to_embeddings=None,
        group_option="ignore",
        dp_composition="snomed_all",
        dp_type="words",
        path_to_vsa_parser_data=None,
        use_vsa=False,
        path_to_pretrained_embeddings=None,
        path_to_pretrained_atomics=None,
        use_cls=False,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self.rotary_value = rotary_value
        self.layer_norm_position = layer_norm_position
        self.freeze_word_embeddings = freeze_word_embeddings
        self.normalize_word_embeddings = normalize_word_embeddings
        self.apply_ffn_to_embeddings = apply_ffn_to_embeddings
        self.group_option = group_option
        self.dp_composition = dp_composition
        self.dp_type = dp_type
        self.path_to_vsa_parser_data = path_to_vsa_parser_data
        self.use_vsa = use_vsa
        self.path_to_pretrained_embeddings = path_to_pretrained_embeddings
        self.path_to_pretrained_atomics = path_to_pretrained_atomics
        self.use_cls = use_cls


class BerthaOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            dynamic_axis = {0: "batch", 1: "sequence"}
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),
                ("attention_mask", dynamic_axis),
                ("token_type_ids", dynamic_axis),
            ]
        )