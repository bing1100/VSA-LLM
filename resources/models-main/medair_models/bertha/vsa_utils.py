import torch
import numpy as np
from functools import reduce
import itertools
from torch.fft import rfft, irfft
from torch import nn
from torch.nn import functional as F
import pickle


class VsaDataParser:
    """
    Class to parse ICD to VSA data into a data structure that can easily be read to generate final concept vectors from atomic vector embedding indices.
    """

    def __init__(self, icd_to_vsa_data, tokenizer_vocab):
        """
        Parameters:
        -----------
        icd_to_vsa_data: dict according to format in `006 - ICD to VSA data.ipynb`
        """
        self.vsa_data = icd_to_vsa_data
        self.tokens_to_ids = tokenizer_vocab
        self.num_cvs = max(self.tokens_to_ids.values()) + 1
        self.icd_to_ids = {}
        self._generate_concept_mappings()
    
    def generate_icd_to_ids_mapping(self, group_option="group_vectors", dp_composition="snomed_all", dp_type="words_rv", vec_types="sp_dp"):
        """
        Generates mapping from ICD to VSA formulation using ids

        Parameters:
        -----------
        group_option: str, by default "ignore". Option for how to deal with SNOMED groups in the semantic pointer
            "ignore": Do not use any group vectors, add up all relationship-concept pairs in all gorups
            "group_zero": Only use relationship-concept pairs in group 0
            "isA": Only use relationship-concept pairs with an isA relationship
            "group_vectors": Apply group vectors to the sums of relationship-concept pairs. Also adds cc_identity as the group for dp
        dp_composition: str, by default "snomed_all". Option for the vocabulary to compose the description pointer
            "icd_name": Use words from the ICD concept name
            "snomed_name": Use words from the SNOMED fully specified name (FSN)
            "snomed_all": Use words from the SNOMED fully specified name and synonyms
        dp_type: str, by default "words_rv". Option for formulation of description pointer
            "atomic": Sum together atomic vectors that represent each SNOMED concept the ICD code is related to. Overrides dp_composition
                since atomic vectors do not have any vocabulary options.
            "words": Sum together all words in the vocabulary specified in dp_composition
            "words_rv": Result of "words", convolved with a description relationship vector
        vec_types: str, by default "sp_dp". Option to control which vectors (sematic pointer, description pointer) are used in the VSA
            "sp_dp": Use both semantic pointer and description pointer
            "sp_only": Use only semantic pointer
            "dp_only": Use only description pointer
        
        Returns:
        --------
        icd_to_ids: dict, with keys as string ICD codes "<code>-<version>" (i.e. "0020-9") and values as a dictionary of lists that can parsed into VSA
        vectors or strings. The keys are the ids for the relationship or tuple of (group id, relationship id), and the values are lists of concept ids.
        """
        self.icd_to_ids = {}
        if vec_types not in ["sp_dp", "dp_only", "sp_only"]:
            raise ValueError("'vec_types' must be one of ['sp_dp', 'sp_only', 'dp_only']")
        if "dp" in vec_types:
            self._generate_dp(dp_composition=dp_composition, dp_type=dp_type, group_option=group_option)
        if "sp" in vec_types:
            self._generate_sp(group_option=group_option)       

        return self.icd_to_ids
            
    def _generate_concept_mappings(self):
        """
        Generate mappings from concepts used in self.vsa_data to the indicies that will be used for atomic vectors
        Atomic vectors are needed for:
            vocab words = V
            SNOMED concepts = C
            SNOMED relationships = R
            SNOMED groups = G
            Extra tokens that don't have VSA representations = T
            Plus one extra for the "description" relationship vector
            Plus one extra for circular convolution identity
        total = V + C + R + G + T + 2
        """
        v = set()
        c = set()
        r = set()
        g = set()
        t = set()
        for data in self.vsa_data.values():
            # All vocab items
            for k in ["icd_name", "snomed_names", "snomed_synonyms"]:
                v.update(data[k])
            # SNOMED concepts
            c.update(data["snomed_concepts"])
            # Relationships
            for group, relations in data.get("relationships", {}).items():
                g.update([group])
                relations, concepts = list(zip(*relations))
                r.update(relations)
                c.update(concepts)
        t = set(self.tokens_to_ids.keys()) - set(self.vsa_data.keys())

        N = len(v) + len(c) + len(r) + len(g) + len(t) + 2
        self.num_avs = N

        # Put the vectors in a sensible order for the mapping
        # Group vectors - sort by numeric order
        # Note groups are changed to new key to avoid overlapping with string representation of integers in vocab
        g = sorted([int(group) for group in g])
        g = [f"group_{i}" for i in g]
        # Relationship vectors
        r = sorted(r)
        description_rv = ["description_rv"]
        # SNOMED concepts
        c = sorted(c)
        # Description vocabulary
        v = sorted(v)
        # Extra tokens
        t = sorted(t)
        # Identity
        identity = ["cc_identity"]

        self._v = v
        self._c = c
        self._r = r
        self._g = g
        self._t = t

        self.concepts_to_ids = {}
        self.ids_to_concepts = {}
        for i, concept in enumerate(itertools.chain.from_iterable([g, r, description_rv, identity, c, v, t])):
            self.concepts_to_ids[concept] = i
            self.ids_to_concepts[i] = concept
        # Make cc_identity be -1
        old_id = self.concepts_to_ids["cc_identity"]
        self.concepts_to_ids["cc_identity"] = -1
        del self.ids_to_concepts[old_id]
        self.ids_to_concepts[-1] = "cc_identity"
        # Make sure there were no overlaps
        assert len(self.concepts_to_ids) == N
        assert len(self.ids_to_concepts) == N
    
    def _lookup_relations(self, relations):
        """
        Parameters:
        -----------
        relations: list of tuples, (relation, concept)
        
        Returns:
        --------
        sp: dict, where keys are relations, values are list of concepts
        """
        sp = {}
        for r, c in relations:
            sp.setdefault(self.concepts_to_ids[r], []).append(self.concepts_to_ids[c])
        return sp

    def _lookup_group(self, g):
        return self.concepts_to_ids[f"group_{g}"]

    def _lookup_extra_token(self, t, group_option=None):
        """
        Parameters:
        -----------
        t: string, key for extra token to look up

        Returns:
        d: dict, where keys are identity relations, values are list of single concept
        """
        cc_id = self.concepts_to_ids["cc_identity"]
        if group_option == "group_vectors":
            key = (cc_id, cc_id)
        else:
            key = cc_id
        value = [self.concepts_to_ids[t]]
        return {key: value}

    def _generate_sp(self, group_option="ignore"):
        """Adds semantic pointer ids to self.icd_to_ids"""
        for icd_key in self.tokens_to_ids.keys():
            data = self.vsa_data.get(icd_key, {})
            if data:
                # Do not use any group vectors, just the totality of all relationships and concepts (including repeats)
                if group_option == "ignore":
                    relations = []
                    for r in data.get("relationships", {}).values():
                        relations.extend(r)
                    sp = self._lookup_relations(relations)
                # isA relationships can only be in group 0
                elif group_option == "group_zero" or group_option == "isA":
                    relations = data.get("relationships", {}).get("0", [])
                    if group_option == "isA":
                        relations = [(r, c) for (r, c) in relations if r == "116680003"]  # Keep only isA relationships
                    sp = self._lookup_relations(relations)
                elif group_option == "group_vectors":
                    sp = {}
                    for group, relations in data.get("relationships", {}).items():
                        g = self._lookup_group(group)
                        # Keys of sp are tuple of (g, r)
                        for r, c in self._lookup_relations(relations).items():
                            sp.setdefault((g, r), []).extend(c)
                else:
                    raise ValueError("'group_option' must be one of ['ignore', 'group_zero', 'isA', 'group_vectors']")
                self.icd_to_ids.setdefault(icd_key, {}).update(sp)
            else:
                # Extra token case
                extra_sp = self._lookup_extra_token(icd_key, group_option=group_option)
                self.icd_to_ids.setdefault(icd_key, {}).update(extra_sp)
    
    def _generate_dp(self, dp_type="words_rv", dp_composition="snomed_all", group_option=None):
        """Adds description pointer ids to self.icd_to_ids"""
        for icd_key in self.tokens_to_ids.keys():
            data = self.vsa_data.get(icd_key, {})
            if data:
                dp = {}
                if dp_type == "atomic":
                    
                    concepts = data.get("snomed_concepts", [])
                    dp = {self.concepts_to_ids["cc_identity"]: [self.concepts_to_ids[c] for c in concepts]}
                elif dp_type in ["words", "words_rv"]:
                    if dp_composition == "snomed_name":
                        words = data.get("snomed_names", [])
                    elif dp_composition == "snomed_all":
                        words = data.get("snomed_names", []) + data.get("snomed_synonyms", [])
                        words = list(set(words))  # Remove duplicates between snomed_names and snomed_synonyms
                    elif dp_composition == "icd_name":
                        words = data.get("icd_name", [])
                    else:
                        raise ValueError("'dp_composition' must be one of ['snomed_name', 'snomed_all', 'icd_name']")
                    word_ids = [self.concepts_to_ids[w] for w in words]

                    # Add convolution with description relationship vecor
                    if dp_type == "words_rv":
                        rv = self.concepts_to_ids["description_rv"]
                    else:
                        rv = self.concepts_to_ids["cc_identity"]
                    # Don't create a relation if there is no DP
                    if word_ids:
                        dp = {rv: word_ids}
                else:
                    raise ValueError("'dp_type' must be one of ['atomic', 'words', 'words_rv']")
                if group_option == "group_vectors":
                    g = self.concepts_to_ids["cc_identity"]  # Use identity as the group for dp
                    dp = {(g, k): v for k, v  in dp.items()}
                self.icd_to_ids.setdefault(icd_key, {}).update(dp)
            else:
                # Extra token case
                extra_dp = self._lookup_extra_token(icd_key, group_option=group_option)
                self.icd_to_ids.setdefault(icd_key, {}).update(extra_dp)
    
    @staticmethod
    def _str_cc(x, y):
        """String representation of circular convolution"""
        return f"[({x}) * ({y})]"

    @staticmethod
    def _str_sum(x, y):
        """String representation of sum"""
        return f"{x} + {y}"

    def parse_vsa_to_str(self, vsa):
        """
        Represent VSA index formulation as a string of atomic vectors with + and * operations
        To parse, we sum over lists, and take product over tuples
        """
        if isinstance(vsa, list):
            # Sum over lists
            return reduce(self._str_sum, map(self.parse_vsa_to_str, vsa))
        if isinstance(vsa, dict):
            # Product on dict
            res = []
            for k, v in vsa.items():
                res.append(self._str_cc(self.parse_vsa_to_str(k), self.parse_vsa_to_str(v)))
            return reduce(self._str_sum, res)
        if isinstance(vsa, tuple):
            # Combine g * r relationships
            return reduce(self._str_cc, map(self.parse_vsa_to_str, vsa))
        if isinstance(vsa, int):
            # Lookup individual elements
            return self.ids_to_concepts[vsa]

    def build_av_cv_mapping(self, mapping):
        """
        Restructure a VSA mapping to give lists of ([source indices], [dest indicies]) for each relationship, where sources are atomic vector
        indices and destinations are concept vector indicies. 
            E.g.
            m = {
                r1: [
                    ([s1, s2, s3], [d1, d2, d3]),
                    ([s4, s5, s6], [d4, d5, d6])
                ],
                r2: [
                    ([s7, s8, s9], [d7, d8, d9])
                ]
                }
            This is so we can perform the VSA operations as CV = (rel1 * av1) + (rel1 * av2) + (rel2 * av3) + ...
            -> CV[dest_ij] += Rel_i * AV[src_j] 

            Because each concept vector may require a different number of atomic vectors, this lookup array would 
            be jagged, so we must store as a list of lists.
        
        Parameters:
        -----------
        mapping: dict, ICD to id mapping from self.generate_icd_to_ids_mapping()

        Returns:
        --------
        av_cv_mapping: dict, resturctured mapping with lists of ([atomic vector indices], [concept vector indices]) for each relationship.
        """
        av_cv_mapping = {}

        # Set of all possible relationships
        relations = set()
        for vsa in mapping.values():
            relations.update(list(vsa.keys()))
        relations = sorted(relations)

        for r in relations:
            tmp = {}
            for icd_key, vsa in mapping.items():
                dest = self.tokens_to_ids[icd_key]  # Index in concept vector matrix
                src = vsa.get(r, [])  # Index in atomic vector matrix
                # Flatten out so that we can perform as many index lookups as possible in one sweep
                # e.g. j=0 will lookup the first atomic vector in ALL concept vectors, j=1 will lookup second AV in ALL concept vectors, if it exists
                for j, s in enumerate(src):
                    tmp.setdefault(j, []).append((s, dest))
            # Remap tmp to lists of src and dest indices
            for _, sd in tmp.items():
                src, dest = list(zip(*sd))
                src, dest = map(list, (src, dest))
                # Each relationship has a list of [source ind -> dest ind] where each source, dest are lists of indices
                # The length of this list depends on the maximum number of times this relationship needs to be used for one concept
                av_cv_mapping.setdefault(r, []).append([src, dest])
        return av_cv_mapping

    def get_cv_gen_args(self, group_option="ignore", dp_composition="snomed_all", dp_type="words_rv", vec_types="sp_dp"):
        """
        Generates arguments for FastCVGen constructor

        Parameters:
        -----------
        group_option: str, by default "ignore". Option for how to deal with SNOMED groups in the semantic pointer
            "ignore": Do not use any group vectors, add up all relationship-concept pairs in all gorups
            "group_zero": Only use relationship-concept pairs in group 0
            "isA": Only use relationship-concept pairs with an isA relationship
            "group_vectors": Apply group vectors to the sums of relationship-concept pairs. Also adds cc_identity as the group for dp
        dp_composition: str, by default "snomed_all". Option for the vocabulary to compose the description pointer
            "icd_name": Use words from the ICD concept name
            "snomed_name": Use words from the SNOMED fully specified name (FSN)
            "snomed_all": Use words from the SNOMED fully specified name and synonyms
        dp_type: str, by default "words_rv". Option for formulation of description pointer
            "atomic": Sum together atomic vectors that represent each SNOMED concept the ICD code is related to. Overrides dp_composition
                since atomic vectors do not have any vocabulary options.
            "words": Sum together all words in the vocabulary specified in dp_composition
            "words_rv": Result of "words", convolved with a description relationship vector
        vec_types: str, by default "sp_dp". Option to control which vectors (sematic pointer, description pointer) are used in the VSA
            "sp_dp": Use both semantic pointer and description pointer
            "sp_only": Use only semantic pointer
            "dp_only": Use only description pointer
        
        Returns:
        --------
        av_cv_mapping: dict, mapping from atomic vector indices to concept vector indicies per each relationship
        num_avs: int, number of atomic vectors needed for the embedding
        num_cvs: int, number of concept vectors needed for the embedding
        padding_av_idx: int, index of the padding vector in the atomic vector embedding
        """
        icd_to_ids = self.generate_icd_to_ids_mapping(group_option, dp_composition, dp_type, vec_types)
        av_cv_mapping = self.build_av_cv_mapping(icd_to_ids)
        padding_av_idx = self.concepts_to_ids["[PAD]"]
        return av_cv_mapping, self.num_avs, self.num_cvs, padding_av_idx


class FastCVGen(torch.nn.Module):
    def __init__(self, av_cv_mapping, num_atomic_vectors, num_concept_vectors, padding_av_idx, config, cc_identity=-1):
        """
        Use VsaParser.get_cv_gen_args() as a convenience function to get arguments for this constructor

        Parameters:
        -----------
        av_cv_mapping: dict, mapping from atomic vector indices to concept vector indicies per each relationship
        num_avs: int, number of atomic vectors needed for the embedding
        num_cvs: int, number of concept vectors needed for the embedding
        padding_av_idx: int, index of the padding vector in the atomic vector embedding
        config: BerthaConfig, configuration object
        cc_identity: int, by default -1. The index in av_cv_mapping that represents the circular convolution identity
        """
        super().__init__()
        self.av_cv_mapping = av_cv_mapping
        self.num_atomic_vectors = num_atomic_vectors
        self.num_concept_vectors = num_concept_vectors
        self.cc_identity = cc_identity
        self.padding_av_idx = padding_av_idx

        self.normalize = config.normalize_word_embeddings
        self.embedding_dim = config.hidden_size
        self.initializer_range = config.initializer_range
        self.use_cls = config.use_cls
        self.path_to_pretrained_atomics = config.path_to_pretrained_atomics

        if config.use_cls == 'cat':
            assert config.path_to_pretrained_atomics is not None, "Cannot use VSA cat with pretrained atomics"
            self.embedding_dim = int(config.hidden_size/2)
            cls_tokens = torch.normal(mean=0.0, std=self.initializer_range, size=(self.num_concept_vectors, self.embedding_dim))
            self.cls_tokens = nn.Parameter(cls_tokens, requires_grad=True)

        if config.use_cls == 'add':
            cls_tokens = torch.normal(mean=0.0, std=self.initializer_range, size=(self.num_concept_vectors, self.embedding_dim))
            self.cls_tokens = nn.Parameter(cls_tokens, requires_grad=True)

        with torch.no_grad():
            self._init_atomic_vectors()

    def _init_atomic_vectors(self):
        if self.path_to_pretrained_atomics is not None:
            avs = torch.load(self.path_to_pretrained_atomics)
            assert avs.shape == torch.Size([self.num_atomic_vectors, self.embedding_dim])
            assert torch.isclose(avs.mean(), torch.Tensor([0]))
            assert torch.isclose(avs.std(), torch.Tensor([self.initializer_range]))
        else:
            avs = torch.normal(mean=0.0, std=self.initializer_range, size=(self.num_atomic_vectors, self.embedding_dim))
        # Padding atomic vector set to zero
        avs[self.padding_av_idx].zero_()
        self.avs = torch.nn.Parameter(avs, requires_grad=True)
        
        # Circular convolution identity
        identity = torch.zeros(self.embedding_dim, dtype=torch.float32)
        identity[0] = 1
        self.identity = torch.nn.Parameter(rfft(identity), requires_grad=False)

    def forward(self):
        """
        Generates the concept vectors based on mapping from atomic vectors
        Note the convention for variable names: Captial letters refer to variables in the Fourier domain, while lower case refers to signal domain

        Returns:
        --------
        cvs: torch.Tensor, concept vectors to be used for embedding lookup
        """
        self.Avs = rfft(self.avs)
        self.Cvs = torch.zeros(self.num_concept_vectors, self.Avs.shape[1], dtype=self.Avs.dtype, device=self.Avs.device)
        for rel_id, av_cv_inds in self.av_cv_mapping.items():
            if isinstance(rel_id, tuple):  # Group vector case
                g, r = rel_id
                g = self.Avs[g, :] if g != self.cc_identity else self.identity
                r = self.Avs[r, :] if r != self.cc_identity else self.identity
                Rel = g * r
            else:
                Rel = self.Avs[rel_id] if rel_id != self.cc_identity else self.identity
            for av_inds, cv_inds in av_cv_inds:
                self.Cvs[cv_inds] += Rel * self.Avs[av_inds]
        cvs = irfft(self.Cvs)
        if self.normalize:
            norm = torch.linalg.norm(cvs, dim=0)
            cvs = cvs / norm
        if self.use_cls == 'cat':
            cvs = torch.cat((self.cls_tokens, cvs), dim=1)
        if self.use_cls == 'add':
            cvs = self.cls_tokens + cvs
        return cvs
    
    def get_avs(self):
        """
        Retrieve real-valued atomic vector embedding matrix
        
        Returns:
        --------
        avs: torch.tensor, real-valued atomic vectors, detached from the computation graph
        """
        avs = torch.fft.irfft(self.Avs.detach())
        return avs


class VsaEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding_dim = config.hidden_size
        self.vocab_size = config.vocab_size
        self.padding_idx = config.pad_token_id
        self.use_vsa = config.use_vsa
        self.freeze_embeddings = config.freeze_word_embeddings
        self.initializer_range = config.initializer_range
        self.path_to_pretrained_embeddings = config.path_to_pretrained_embeddings

        # Embedding initialization
        if config.path_to_pretrained_embeddings is not None:
            if config.use_vsa:
                raise ValueError("`config.use_vsa` must be set to `False` if using pre-trained embeddings")
            self.embedding_gen = self._load_pretrained_embedding
        elif config.use_vsa:
            if config.path_to_vsa_parser_data is None:
                raise ValueError("`config.path_to_vsa_parser_data` must be passed if using `config.use_vsa = True`")
            if config.tokenizer_vocab is None:
                raise ValueError("`config.tokenizer_vocab` must be passed if using `config.use_vsa = True`")

            with open(config.path_to_vsa_parser_data, 'rb') as f:
                parser_data = pickle.load(f)
            parser = VsaDataParser(parser_data, config.tokenizer_vocab)
            args = parser.get_cv_gen_args(config.group_option, config.dp_composition, config.dp_type)
            self.embedding_gen = FastCVGen(*args, config)
            self.num_avs = args[1]
            self.num_cvs = args[2]
            assert self.num_cvs == self.vocab_size, f"Number of concept vectors ({self.num_cvs}) must equal the vocab size ({self.vocab_size})"
            
        else:
            self.embedding_gen = self._init_random_embedding

        # Generate the embeddings
        with torch.no_grad():
            embedding = self.embedding_gen()

        # Freezing options
        if self.freeze_embeddings:
            self.weight = nn.Parameter(embedding, requires_grad=False)
        elif self.use_vsa:
            # VSA not frozen, generate at runtime into a buffer
            self.register_buffer("weight", embedding)
        else:
            self.weight = nn.Parameter(embedding, requires_grad=True)
        return

    def _init_random_embedding(self):
        embedding = torch.normal(mean=0.0, std=self.initializer_range, size=(self.vocab_size, self.embedding_dim))
        embedding[self.padding_idx] = torch.zeros(self.embedding_dim)
        return embedding

    def _load_pretrained_embedding(self):
        embedding = np.load(self.path_to_pretrained_embeddings)
        assert embedding.shape == (self.vocab_size, self.embedding_dim), f"Embeddings must match sizes in config ({self.vocab_size}, {self.embedding_dim})"
        embedding = torch.as_tensor(embedding, dtype=torch.float)
        return embedding

    def convert_vsa_to_embedding(self, freeze_embeddings=False):
        """
        Configures self to convert VSA pipeline to a regular embedding matrix
        Sets self.weight to be a nn.Parameter which may store gradients

        Parameters:
        -----------
        freeze: bool, by default False, whether to freeze the resulting embedding matrix
        """
        if self.use_vsa is False:
            raise ValueError("self.use_vsa must be True to perform VSA conversion")
        
        with torch.no_grad():
            embedding = self.embedding_gen()
        self.weight = nn.Parameter(embedding, requires_grad=(not freeze_embeddings))
        self.use_vsa = False
        self.freeze_embeddings = freeze_embeddings

    def forward(self, input_ids):
        if self.use_vsa and not self.freeze_embeddings:
            # Generate embeddings each forward pass in non-frozen VSA case
            embedding = self.embedding_gen()
            self.weight = embedding
        else:
            embedding = self.weight
        return F.embedding(input_ids, embedding, padding_idx=self.padding_idx)
