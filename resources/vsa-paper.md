%% The first command in your LaTeX source must be the \documentclass command.
%%
%% Options:
%% twocolumn : Two column layout.
%% hf: enable header and footer.
\documentclass[
twocolumn,
% hf,
]{ceurart}

%%
%% One can fix some overfulls
\sloppy

\input{math_commands.tex}



% \usepackage{algorithm}
% \usepackage{algpseudocode}
% \usepackage{amsmath}
% \usepackage{amssymb}
% \usepackage{mathrsfs}
% \usepackage{multirow}
% \usepackage[pdftex]{graphicx}

%%
%% Minted listings support 
%% Need pygment <http://pygments.org/> <http://pypi.python.org/pypi/Pygments>
% \usepackage{hyperref}
% \usepackage{url}

\usepackage{listings}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{amsfonts,bm}
\usepackage{mathrsfs}
\usepackage{multirow}
%\usepackage[pdftex]{graphicx}
\usepackage{graphicx}

\usepackage{hyperref}
\hypersetup{hidelinks=true}
\usepackage{textcomp}
%% auto break lines
\lstset{breaklines=true}

%%
%% end of the preamble, start of the body of the document source.
\begin{document}

%%
%% Rights management information.
%% CC-BY is default license.
\copyrightyear{2024}
\copyrightclause{Copyright for this paper by its authors.
  Use permitted under Creative Commons License Attribution 4.0
  International (CC BY 4.0).}

%%
%% This command is for the conference information
\conference{KiL'24: Workshop on Knowledge-infused Learning co-located with 30th ACM KDD Conference,
August 26, 2024, Barcelona, Spain}

%%
%% The "title" command
\title{Encoding Medical Ontologies With Holographic Reduced Representations for Transformers}

% \tnotemark[1]
% \tnotetext[1]{You can use this document as the template for preparing your
%   publication. We recommend using the latest version of the ceurart style.}

%%
%% The "author" command and its associated commands are used to define
%% the authors and their affiliations.
\author[1]{Bing Hu}[%
email=bingxu.hu@uwaterloo.ca
]
\cormark[1]
\address[1]{University of Waterloo,
  Ontario, Canada}
\address[2]{McMaster University,
  Ontario, Canada}

\author[1]{Trevor Yu}[%
]

\author[1]{Tia Tuinstra}[%
]

\author[1]{Ryan Rezai}[%
]

\author[1]{Harshit Bokadia}[%
]

\author[1]{Rachel DiMaio}[%
]

\author[1]{Thomas Fortin}[%
]

\author[1,2]{Brian Vartian}[%
]

\author[1]{Bryan Tripp}[%
]

%% Footnotes
\cortext[1]{Corresponding author.}
% \fntext[1]{These authors contributed equally.}

%%
%% The abstract is a short summary of the work to be presented in the
%% article.
\begin{abstract}
% Methods from natural language processing (NLP) are increasingly applied to electronic health records (EHRs), with previous work having applied transformers to sequences of medical codes.
% The ability to encode meaningful structure into deep learning models opens up the potential for incorporating prior knowledge, particularly in fields where domain-specific information is of great importance.
% However, transformer
Transformer models trained on NLP tasks with medical codes often have randomly initialized embeddings that are then adjusted based on training data. 
For terms appearing infrequently in the dataset, there is little opportunity to improve these representations and learn semantic similarity with other concepts. 
Medical ontologies represent many biomedical concepts and define a relationship structure between these concepts, making ontologies a valuable source of domain-specific information. 
% One of the ongoing challenges of deep learning is finding methods to incorporate this domain knowledge into models.
Holographic Reduced Representations (HRR) are capable of encoding ontological structure by composing atomic vectors to create structured higher-level concept vectors. 
% Deep learning models can further process these structured vectors without needing to learn the ontology from training data. 
We developed an embedding layer that generates concept vectors for clinical diagnostic codes by applying HRR operations that compose atomic vectors based on the SNOMED CT ontology. 
This approach allows for learning %to update
the atomic vectors while maintaining structure in the concept vectors.
We trained a Bidirectional Encoder Representations from the Transformers (BERT) model to process sequences of clinical diagnostic codes and used the resulting HRR concept vectors as the embedding matrix for the model.
% The model was first pre-trained on a masked-language modeling (MLM) task before being fine-tuned for mortality and disease prediction tasks. 
% The HRR-based approach improved performance on the pre-training and fine tuning tasks compared to standard transformer embeddings. 
The HRR-based approach introduced interpretable structure into code embeddings while maintaining or modestly improving performance on the masked language modeling (MLM) pre-training task (particularly for rare codes) as well as the fine-tuning tasks of mortality and disease prediction.
% (particularly for patients with many rare codes). 
% This method also supports explainability by separating representations of code-frequency, ontological information, and description words.  
% This is the first time HRRs have been used to produce structured embeddings for transformer models and we find that t
This approach also better maintains semantic similarity between medically related concept vectors, due to both shared atomic vectors and disentangling of code-frequency information. 
% and allows better representations to be learned for rare codes in the dataset. 
% , as rare codes are composed of elements that are shared with more common codes.
% This has great significance for future applications of deep networks to electronic health records, especially when limited medical datasets are available and clinical coding practices may differ across health domains.
%, which may be of great benefit to model performance and explainability, particularly for concepts that appear infrequently in training data
\end{abstract}

%%
%% Keywords. The author(s) should pick words that accurately describe
%% the work being presented. Separate the keywords with commas.
\begin{keywords}
  Deep Learning \sep
  Ontology \sep
  Knowledge-Integration
\end{keywords}

%%
%% This command processes the author and affiliation and title
%% information and builds the first part of the formatted document.
\maketitle

\section{Introduction}

\label{sec:introduction}

Transformers \cite{vaswani2017attention} jointly optimize high-dimensional vector embeddings that represent input tokens, and a network that contextualizes and transforms these embeddings to perform a task. 
Originally designed for natural language processing (NLP) tasks, transformers are now widely used with other data modalities. 
In medical applications, one important modality consists of medical codes that are extensively used in electronic health records (EHR). 
A prominent example in this space is Med-BERT \cite{med-bert}, which consumes a sequence of diagnosis codes. Tasks that Med-BERT and other EHR-transformers perform include disease and mortality prediction. 

% As the names imply, the former is the task of predicting which disease a patient will be diagnosed with next while the latter is the task of predicting the risk of death for a patient in a given window of time. 

% Although ambitions exist for fully autonomous AI systems \cite{Kitano_2016}, many ambitions for AI systems in science, life sciences, and medical applications remain semi-automated with a human-in-the-loop \cite{ai4science,HITLAIReview,Min2016DeepLI}. For such a collaborative AI system to work effectively, ontologies can be used to communicate domain knowledge to the machine in a suitably precise form and to provide a common declarative knowledge representation \cite{dash2021incorporating, HITLAIReview}. 

% In AI systems for medical applications, a doctor-in-the-loop workflow is critical as human expertise and experience are often necessary to make use of the complex or sparse data that is prevalent in medical applications \cite{HITLAIReview, HITLsurvey2022}. 

Deep networks have traditionally been alternatives to symbolic artificial intelligence with different advantages \cite{learnhrr}. 
Deep networks use real-world data effectively, but symbolic approaches have completive properties, such as better transparency and capacity for incorporating structured information, inspiring many efforts to combine the two approaches in neuro-symbolic systems \cite{sarker2021neuro}. 
Additional transparency and ability to incorporate structured information are potential benefits of symbolic approaches in medical applications
% , such as logic-based programming, and clinical decision support, for medical applications potentially over large language models 
\cite{ramgopal2023artificial}. 
Standard large language models (LLMs) can be prone to biases in the training data, such as frequency bias, which can result in medical misinformation and potentially clinical harm \cite{freqbias,biderman2023pythia,singhal2023large}. 
% By incorporating structured information, LLM-prone biases, such as frequency bias, are mitigated in symbolic approaches and methods.
% Such limitations have inspired many efforts to combine the two approaches in neuro-symbolic systems \cite{sarker2021neuro}. 

Here we use a novel neuro-symbolic medical transformer architecture incorporating structured knowledge from an authoritative medical ontology into the embeddings. 
Specifically, we use vector-symbolic holographic reduced representations (HRRs) \cite{platehrr} to produce composite medical-code embeddings and backpropagate through the architecture to optimize the embeddings of atomic concepts. 
This approach produces optimized medical code embeddings with an explicit structure that incorporates medical knowledge. 

We test our method, Holographic Reduced Representation Bi-directional Encoder Representations from Transformers (HRRBERT), on the Medical Information Mart for Intensive Care (MIMIC)-IV  dataset \cite{MIMIC-IV-v2.0} and show improvements in both pre-training and fine-tuning tasks. 
We also show that our embeddings of ontologically similar rare medical codes have high cosine similarity, in contrast with embeddings that are learned in the standard way.
Finally, we investigate learned representations of medical-code frequency, in light of recent demonstration of frequency bias in EHR-transformers \cite{freqbias}. 


% In our paper, we propose to study the viability of such a neural-symbolic approach in integrating medical ontologies into medical transformer models. The purpose of our work is to investigate the learning dynamics of using HRR as a differentiable component applied to the Transformer Architecture in a hybrid neural-symbolic approach. As a conclusion of our investigations, we present key contributions in:
We contribute: 
\begin{itemize}
    \item A novel neuro-symbolic architecture, HRRBERT, that combines vector-symbolic embeddings with the BERT LLM architecture, leading to better performance in medical tasks.
    % \item Efficient vector-symbolic binding operations that leverage GPU-optimized CUDA computations.
    \item Efficient construction of vector-symbolic embeddings that leverage PyTorch autograd on GPUs.
    \item Optimized medical-code embeddings that better respect semantic similarity of medical terminology than standard embeddings for infrequently used codes.   
\end{itemize}

We focus here on processing medical codes, but our methods would extend naturally to foundation models that combine medical codes and natural language. Specifically, the trained atomic vectors of our vector-symbolic embeddings could share a dictionary with language embeddings, so that training of each could improve the representation of the other. 

\subsection{Background and Related Works}

The Vector-Symbolic Architectures (VSA) approach 
% also called hyperdimensional computing (HD) \cite{gehd}, 
is a computing paradigm that relies on high dimensionality and randomness to represent concepts as unique vectors in a high dimensional space \cite{Kanerva2009HyperdimensionalCA}. VSAs create and manipulate distributed representations of concepts by combining base vectors with bundling, binding, and permutation algebraic operators \cite{gayler2004vector}. For example, a scene with a red box and a green ball could be described with the vector SCENE$=$RED$\otimes$BOX$+$GREEN$\otimes$BALL, where $\otimes$ indicates binding, and $+$ indicates bundling. The atomic concepts of RED, GREEN, BOX, and BALL are represented by base vectors, which are typically random. VSAs also define an inverse operation that allows the decomposition of a composite representation. For example, the scene representation could be queried as SCENE$\otimes$BOX$^{-1}$. This should return the representation of GREEN or an approximation of GREEN that is identifiable when compared to a dictionary. In a VSA, the similarity between concepts can be assessed by measuring the distance between the two corresponding vectors.

VSAs were proposed to address challenges in modelling cognition, particularly language \cite{gayler2004vector}. However, VSAs have been successfully applied across a variety of domains and modalities outside of the area of language as well, including in vision \cite{Neubert_Schubert_2021, Neubert_Schubert_Schlegel_Protzel_2021}, biosignal processing \cite{Rahimi_biosignal}, and time-series classification \cite{Schlegel_Neubert_Protzel_2022}. Regardless of the modality or application, VSAs provide value by enriching vectors with additional information, such as spatial semantic information in images and global time encoding in time series.

An early VSA framework was Smolensky's Tensor Product Representation \cite{SMOLENSKY1990159}, which addressed the need for compositionality, but suffered from exploding model dimensionality. The VSA framework introduced by Plate, Holographic Reduced Representations (HRR), improved upon Smolensky's by using circular convolution as the binding operator \cite{platehrr}. 
Circular convolution keeps the output in the same dimension, solving the problem of exploding dimensionality. 

% With HRR, Plate shows that it is possible to encode a conceptual symbolic meaning into the high-dimensional vector representations, including for language processing or reasoning systems \cite{platehrr}.

In the field of deep learning, HRRs have been used in previous work to recast self-attention for transformer models \cite{alam2023recasting}, to improve the efficiency of neural networks performing a multi-label classification task by using an HRR-based output layer \cite{learnhrr}, and as a learning model itself with a dynamic encoder that is updated through training \cite{efficienthdd}. In all of these works, the efficiency and simple arithmetic of HRRs are leveraged. Our work differs in that we also leverage the ability of HRRs to create structured vectors to represent complex concepts as inputs to a transformer model.

VSAs such as HRRs can effectively encode domain knowledge, including complex concepts and the relationships between them. For instance, Nickel et al. \cite{Nickel_Rosasco_Poggio_2015} propose holographic embeddings that make use of VSA properties to learn and represent knowledge graphs. Encoding domain knowledge is of interest in the field of deep learning, as it could improve, for example, a deep neural network's ability to leverage human knowledge and to communicate its results within a framework that humans understand \cite{Dash_2021}. Ontologies are a form of domain knowledge incorporated into machine learning models to use background knowledge to create embeddings with meaningful similarity metrics and for other purposes  \cite{semsimontsurvey}. 
% In our work, we use the SNOMED CT medical ontology and HRRs to encode domain knowledge in trainable embeddings for a transformer model.
In our work, we use HRRs to encode domain knowledge in trainable embeddings for a transformer model. The domain knowledge we use comes from the Systematized Nomenclature of Medicine Clinical Terms (SNOMED CT), which is a widely used clinical ontology system that includes definitions of relationships between clinical concepts \cite{snomed}. 
% Additional information on SNOMED CT is provided in Appendix \ref{app:ont}.

To the best of our knowledge, HRRs have not been used before as embeddings for transformer models. Transformer models typically use learned embeddings with random initializations \cite{vaswani2017attention}. However, in the context of representing ontological concepts, using such unstructured embeddings can have undesirable effects. One problem is the inconsistency between the rate of co-occurrence or patterns of occurrence of medical concepts and their degree of semantic similarity described by the ontology. For example, the concepts of ``Type I Diabetes'' and ``Type II Diabetes'' are mutually exclusive in EHR data and do not follow the same patterns of occurrence due to differences in pathology and patient populations \cite{ijcai2019p641}. The differences in occurrence make it difficult for a transformer model to learn embeddings with accurate similarity metrics. The concepts should have relatively high similarity according to the ontology. They both share a common ancestor of ``Diabetes Mellitus,'' they are both metabolic disorders that affect blood glucose levels, and they can both lead to similar health outcomes. Song et al. \cite{ijcai2019p641} seeks to address this type of inconsistency by training multiple ``multi-sense'' embeddings for each non-leaf node in an ontology's knowledge graph via an attention mechanism. However, the ``multi-sense'' embeddings do not address the learned frequency-related bias that also arises from the co-occurrence of concepts. Frequency-related bias raises an explainability issue, as it leads to learned embeddings that do not reflect true similarity relationships between concepts, for example, as defined in an ontology, but instead reflect the frequency of the concepts in the dataset \cite{freqbias}. This bias particularly affects codes that are used less frequently.

% Hyphen added to high-dim-ensional for formatting error
Our proposed approach, HRRBERT, uses the structure from SNOMED CT to represent thousands of concepts with high-dim-ensional vectors such that each vector reflects a particular clinical meaning and can be compared to other vectors using the HRR similarity metric, cosine similarity. It also leverages the computing properties of HRRs to provide 
% a fault-tolerant and robust framework 
structured embeddings for a LLM that supports optimization through backpropagation.

\section{Methods}

\subsection{MIMIC-IV Dataset}

The data used in this study was derived from the Medical Information Mart for Intensive Care (MIMIC) v2.0 database, which is composed of de-identified EHRs from in-patient hospital visits between 2008 and 2019 \cite{MIMIC-IV-v2.0}. MIMIC-IV is available through PhysioNet \cite{PhysioNet}.
% As in \cite{freqbias}, 
% We obtained the sequence of ICD-9 and ICD-10 diagnostic codes from the patient histories and kept patients with at least one ICD code to construct our dataset. 
We used the ICD-9 and ICD-10 diagnostic codes from the \textit{icd\_diagnosis} table from the MIMIC-IV \textit{hosp} module. We filtered patients who did not have at least one diagnostic code associated with their records. Sequences of codes were generated per patient by sorting their hospital visits by time. Within one visit, the order of codes from the MIMIC-IV database was used, since it represents the relative importance of the code for that visit. Each unique code was assigned a token.
In total, there were 189,980 patient records in the dataset. We used 174,890 patient records for pre-training, on which we performed a 90--10 training-validation split. We reserved 15k records for fine-tuning tasks.

\subsection{Model Architecture}\label{app:model}

We utilized a BERT-base model architecture with a post-layer norm position and a sequence length of 128 ICD codes \cite{devlin-bert}. A custom embedding class was used to support the functionality required for our HRR embeddings. We adapted the BERT segment embeddings to represent groups of codes from the same hospital visit, using up to 100 segment embeddings to encode visit sequencing. An embedding dimension of $d=768$ was used, and all embeddings were initialized from $\vx \sim \mathcal{N}_d(0, 0.02)$, as in \cite{devlin-bert}, including the atomic vectors for HRR embeddings. 
% A dropout rate of 0.1 was used in the model. We used Adam with a learning rate of 1e-4, $\beta_1 = 0.9$, and $\beta_2 = 0.999$. For pre-training, the learning rate was scheduled with 10\% warm-up steps and cosine decay, and no weight decay was applied. Pre-training lasted 100 epochs with a batch size of 96. 
Fine-tuning used a constant learning rate schedule with a weight decay of 4e-6. Fine-tuning lasted 10 epochs with a batch size of 80. 

\subsection{Encoding SNOMED Ontology with HRR Embeddings} \label{sec:hrr-embeddings}

In this section, we detail the methodologies of constructing vector embeddings for ICD disease codes using HRR operations based on the SNOMED CT structured clinical vocabulary. We first describe our mapping from ICD concepts to SNOMED CT terms. Next, we define how the atomic symbols present in the SNOMED CT ontology are combined using HRR operations to construct concept vectors for the ICD codes. Finally, we describe our method to efficiently compute the HRR embedding matrix using default PyTorch operations that are compatible with autograd.

\subsubsection{Mapping ICD to SNOMED CT Ontology}
Our data uses ICD-9 and ICD-10 disease codes while our symbolic ontology is defined in SNOMED CT, so we required a mapping from the ICD to the SNOMED CT system to build our symbolic architecture. We used the SNOMED CT International Release from May 31, 2022 \cite{snomed} and only included SNOMED CT terms that were active at the time of that release. While SNOMED publishes a mapping tool from SNOMED CT to ICD-10, a majority of ICD-10 concepts have one-to-many mappings in the ICD-to-SNOMED CT direction \cite{snomed-to-icd10}. To increase the fraction of one-to-one mappings, we used additional published mappings from the Observational Medical Outcomes Partnership (OMOP) \cite{omop}, mappings from ICD-9 directly to SNOMED CT \cite{icd9-to-snomed}, and mappings from ICD-10 to ICD-9 \cite{icd10-to-icd9}. 
% Specific details on how these mappings were used can be found in Appendix \ref{map_app}.

Notably, after excluding ICD codes with no active SNOMED CT mapping, 671 out of the 26,164 unique ICD codes in the MIMIC-IV dataset were missing mappings. When those individual codes were removed, a data volume of 4.62\% of codes was lost. This removed 58 out of 190,180 patients from the dataset, as they had no valid ICD codes in their history. Overall, the remaining 25,493 ICD codes mapped to a total of 12,263 SNOMED CT terms.

\subsubsection{SNOMED CT vector symbolic architecture}
Next, we define how the contents of the SNOMED CT ontology were used to construct a symbolic graph to represent ICD concepts. For a given SNOMED CT term, we used its descriptive words and its relationships to other SNOMED CT terms. A relationship is defined by a relationship type and a target term. In total, there were 13,852 SNOMED CT target terms and 40 SNOMED CT relationship types used to represent all desired ICD concepts. In the ontology, many ICD concepts share SNOMED CT terms in their representations.

The set of relationships was not necessarily unique for each SNOMED CT term. To add more unique information, we used a term's ``fully specified name'' and any ``synonyms'' as an additional set of words describing that term. We set all text to lowercase, stripped punctuation, and split on spaces to create a vocabulary of words. We removed common English stopwords from a custom stopword list that was collected with assistance from a medical physician. The procedure resulted in a total of 8833 vocabulary words.

Overall, there were a total of 22,725 ``atomic'' symbols for the VSA which included the SNOMED CT terms, relationships, and the description vocabulary. Each symbol was assigned an ``atomic vector''. We built a ``concept vector'' for each of the target 25,493 ICD codes using HRR operations to combine atomic vectors according to the SNOMED CT ontology structure.

To build a $d$-dimensional concept vector for a given ICD concept, we first considered the set of all relationships that the concept maps to. We used the HRR operator for binding, circular convolution, to combine vectors representing the relationship type and destination term and defined the concept vector to be the bundling of these bound relationships. For the description words, we bundled the vectors representing each word together and bound this result with a new vector representing the relationship type ``description,'' as shown in \autoref{eq:hrr}. 
% Although there is no encoding for the group with the description words in SNOMED CT, 
% We used the identity vector for circular convolution, $\ve^{(1)}$, as the vector representing the group for description words.
\begin{equation}
\label{eq:hrr}
    \vx_{\text{ICD concept}} = \sum_{\text{SNOMED CT}} \vx_{\text{rel}} \circledast \vx_{\text{term}} + \sum_{\text{words}} \vx_{\text{desc}} \circledast \vx_{\text{word}}
\end{equation}


% For simplicity, we considered the binding of groups and relationship types within SNOMED CT to be one relationship vector. This allowed us to combine our description formulation with the SNOMED CT relationships in a common structure and improve the efficiency of computation. 
Formally, let $\sA : \{1, 2, ..., N_a\}$ be the set of integers enumerating the unique atomic symbols for SNOMED CT terms and description words. Let $\sB : \{ 1, 2, ..., N_r\}$ be the set of integers enumerating unique relationships for SNOMED CT terms, including the description relationship and the binding identity. Let $\sD : \{ 1, 2, ..., N_c\}$ be the set of integers enumerating the ICD-9 and ICD-10 disease concepts represented by the VSA.

 $\sA$ has an associated embedding matrix $\mA \in \R^{N_a \times d}$, where atomic vector $\va_k = \mA_{[k, :]},\ k\in\sA$ is the $k$-th row the embedding matrix. Similarly, there is relationship embedding matrix, $\mR \in \R^{N_r \times d}$ and $\vr_j = \mR_{[j, :]},\  j \in \sB$; and an ICD concept embedding matrix, $\mC \in \R^{N_c \times d}$ and $\vc_i = \mC_{[i, :]},\ i \in \sD$. We described the VSA with the formula in \autoref{eq:concept_vec}, where $\mathcal{G}_i$ is a graph representing the connections between ICD concept $i$ to atomic symbols $k$ by relationship $j$. 
\begin{equation}
\label{eq:concept_vec}
\vc_i = \sum_{(j, k) \in \mathcal{G}_i}  \vr_j \circledast \va_k
\end{equation}

% Additional details on model architecture are provided in Appendix \ref{app:model}. 
Additional details on how to efficiently use PyTorch autograd to learn through these HRR operations are provided in Appendix \ref{app:learn-hrr}.

\subsubsection{Embedding Configurations}
We call our method of constructing embeddings for ICD codes purely from HRR representations ``HRRBase'' and the standard method of creating transformer token embeddings from random vectors ``unstructured''. While the HRRBase configuration enforces the ontology structure, we wondered whether it would be too rigid and have difficulty representing information not present in SNOMED CT. 
As dataset frequency information for ICD medical codes is not present in the HRR structure, we tried adding an embedding that represented the empirical frequency of that ICD code in the dataset. We also tried adding fully learnable embeddings with no prior structure.

% \paragraph{Frequency Embeddings:} 
Given the wide range of ICD code frequencies in MIMIC, we log-transformed the empirical ICD code frequencies, and then discretized the resulting range. For our HRRFreq configuration, we used the sinusoidal frequency encoding as in \cite{vaswani2017attention} to encode the discretized log-frequency information. The frequency embeddings were normalized before being summed with the HRR embedding vectors. 

% \paragraph{Learnable Embeddings:} 
We defined two additional configurations in which a standard embedding vector was integrated with the structured HRR concept vector. With ``HRRAdd'', a learnable embedding was added to the concept embedding, HRRAdd = $\mC + \mL_{\text{add}}, \mL_{\text{add}} \in \R^{N_c \times d}$. However, this roughly doubled the number of learnable parameters compared to other formulations. 

With ``HRRCat'', a learnable embedding of dimension $d/2$ was concatenated with the HRR concept embedding of dimension $d/2$. This keeps the total number of learnable parameters roughly the same as the unstructured configuration (25,493 $d$-dimensional vectors) and the HRRBase configuration (22,725 $d$-dimensional vectors). The final embedding matrix was defined as HRRCat = $[\mC \ \mL_{\text{cat}}]$, where $\mC, \mL_{\text{cat}} \in R^{N_c \times d/2}$.

\subsection{Experiments}
We pre-trained the unstructured, HRRBase, HRRCat, and HRRAdd embedding configurations of HRRBERT on the masked language modelling (MLM) task, for 3 trials each. For each of the 3 pre-trained models, 10 fine-tuning trials were conducted for a total of 30 trials per fine-tuning task. The best checkpoint from the 10 epochs of fine-tuning was saved based on validation performance. A test set containing 666 patient records was used to evaluate each of the fine-tuned models for both mortality and disease prediction. We report accuracy, precision, recall, and F1 scores averaged over the 30 trials for the fine-tuning tasks.


\section{Experimental Results}
\subsection{Pre-training}

\begin{figure}[h]
\begin{center} 
\includegraphics[width=1\linewidth]{figures/pretraining.png} 
% \fbox{\rule[-.5cm]{0cm}{4cm} \rule[-.5cm]{4cm}{0cm}}
\end{center}
\caption{Pre-training validation set evaluation results for different configurations}\label{fig:pretraining}
\end{figure}

MLM accuracy is evaluated on a validation set over the course of pre-training. Pre-training results for different configurations are shown in \autoref{fig:pretraining}. The pre-training results are averaged over 3 runs for each of the configurations except for HRRFreq where only 1 model run was completed.

The baseline of learned unstructured embeddings has a peak pre-training validation performance of around 33.4\%. HRRBase embeddings perform around 17\% worse compared to the baseline of learned unstructured embeddings. We hypothesize that this decrease in performance is due to a lack of embedded frequency information in HRRBase compared to learned unstructured embeddings. HRRFreq (which combines SNOMED CT information with frequency information) has a similar performance compared to unstructured embeddings, supporting this hypothesis. Compared to baseline, %HRRAdd, HRRCat, and HRROverlap
HRRAdd and HRRCat improve pre-training performance by a modest margin of around 2\%. We posit that this almost 20\% increase in performance of HRRCat and HRRAdd over HRRBase during pre-training is partly due to the fully learnable embedding used in HRRCat and HRRAdd learning frequency information. 

\subsection{Fine-tuning}
We fine tuned the networks for mortality prediction and disease prediction. Across metrics and tasks, the best results were often seen in HRRBase (Table \ref{table:finetuneprediction}) with some being statistically significant.
%, but only a few of these results were statistically significant. 

\subsubsection{Mortality Prediction Task} The mortality prediction task is defined as predicting patient mortality within 6 months after the last visit. 
Binary mortality labels were generated by comparing the time difference between the last visit and the mortality date. A training set of 13k patient records along with a validation set of 2k patient records were used to fine-tune each model on mortality prediction. \autoref{table:finetuneprediction} shows the evaluation results of mortality prediction for each of the configurations. We performed a two-sided Dunnett's test to compare our multiple experimental HRR embedding configurations to the control unstructured embeddings, with $p<0.05$ significance level. 
% Levene's test shows that the equal variance condition is satisfied, and the Shapiro-Wilk test suggests normal distributions except for HRRBase precision and F1.
HRRBase embeddings had a significantly greater mean F1-score $(p = 0.043)$ and precision $(p=0.042)$ compared to unstructured embeddings. 
% No comparisons of mean metrics for HRR embeddings were significantly greater than the control.

\begin{table*}[t]
\caption{Finetuning mean test scores and standard deviations for mortality prediction, disease prediction, eICU mortality prediction, and both Really-Out-Of-Distribution (ROOD) Unseen and Overall disease prediction tasks. The best scores are bolded and are underlined if statistically significant.}\label{table:mortality}
\label{table:finetuneprediction}
\begin{center}
\begin{tabular}{ c c c c c c } 
\bf Finetuning Task & \bf Configuration & \bf Accuracy & \bf Precision & \bf Recall & \bf F1-Score \\
\hline
\multirow{2}{5em}{ROOD Unseen} & \multirow{1}{6em}{HRRBase} & \underline{\textbf{94.9}$\pm$1.0} & \underline{\textbf{83.5}$\pm$4.6} & \underline{\textbf{76.8}$\pm$5.1} & \underline{\textbf{79.5}$\pm$4.9} \\ 
& \multirow{1}{6em}{Unstructured} &92.3$\pm$0.3 &46.2$\pm$0.0 &50.0$\pm$0.1 & 48.0$\pm$0.1 \\ 
\hline
\multirow{2}{5em}{ROOD Overall} & \multirow{1}{6em}{HRRBase} & \textbf{81.9}$\pm$0.1 & 78.3$\pm$0.3 & \textbf{75.2}$\pm$0.8 & \textbf{76.4}$\pm$0.5\\ 
& \multirow{1}{6em}{Unstructured} &81.9$\pm$0.2 &\textbf{78.7}$\pm$0.7 &74.4$\pm$1.2&76.0$\pm$0.8 \\ 
\hline
\multirow{4}{5em}{Mortality Prediction} & \multirow{1}{6em}{HRRBase} & \textbf{84.4}$\pm$2.3 & \underline{\textbf{65.8}$\pm$2.0} & 85.6$\pm$2.2 & \underline{\textbf{69.2}$\pm$2.7} \\ 
& \multirow{1}{6em}{HRRAdd} &84.0$\pm$2.2 &65.7$\pm$1.9 &\textbf{85.7}$\pm$2.3 & 68.9$\pm$2.5 \\ 
& \multirow{1}{6em}{HRRCat} &83.9$\pm$2.3 &65.6$\pm$1.7 &84.9$\pm$2.8 & 68.8$\pm$2.5\\ 
& \multirow{1}{6em}{Unstructured} &83.4$\pm$1.9 &64.9$\pm$1.2 &84.6$\pm$2.2 & 67.9$\pm$1.8 \\ 
\hline
\multirow{4}{5em}{Disease Prediction} & \multirow{1}{6em}{HRRBase} & \underline{\textbf{79.9}$\pm$0.5} & \underline{\textbf{73.0}$\pm$1.2} & 67.2$\pm$0.7 & 69.0$\pm$0.6\\ 
& \multirow{1}{6em}{HRRAdd} &79.6$\pm$0.7 &72.6$\pm$1.4 &67.3$\pm$0.9 & 69.0$\pm$0.6\\ 
& \multirow{1}{6em}{HRRCat} &79.6$\pm$0.8 &72.5$\pm$1.7 &67.3$\pm$1.0 & 68.9$\pm$0.8\\ 
& \multirow{1}{6em}{Unstructured} &79.4$\pm$0.5 &72.1$\pm$1.1 &\textbf{67.8}$\pm$1.0&\textbf{69.2}$\pm$0.7 \\ 
\hline
\multirow{3}{5em}{eICU Mortality Prediction} & \multirow{1}{6em}{HRRBase} & \underline{\textbf{68.9}$\pm$1.3} & \textbf{75.0}$\pm$1.8 & \textbf{57.0}$\pm$5.8 & \textbf{64.5}$\pm$3.5 \\ 
& \multirow{1}{6em}{HRRAdd} &68.1$\pm$1.6 &74.0$\pm$2.2 &56.2$\pm$6.8 & 63.6$\pm$3.9 \\ 
& \multirow{1}{6em}{HRRCat} &68.2$\pm$1.2 &73.8$\pm$2.6 &57.0$\pm$7.2 & 64.0$\pm$3.7\\ 
& \multirow{1}{6em}{Unstructured} &68.0$\pm$1.4 &74.0$\pm$2.6 &56.0$\pm$7.2 & 63.4$\pm$3.9 \\
\hline

\end{tabular}
\end{center}
\end{table*}

\subsubsection{Disease Prediction Task} The disease prediction task is defined as predicting 
%the disease chapters as a multi-label for 
which disease chapters were recorded in
the patient's last visit using information from earlier visits. We converted all ICD codes in a patient's last visit into a multi-label binary vector of disease chapters. As there are 22 disease chapters defined in ICD-10, the multi-label binary vector has a size of 22 with binary values corresponding to the presence of a disease in each chapter. A training set of 4.5k patient records along with a validation set of 500 patient records were used to fine-tune each model on this task. \autoref{table:finetuneprediction} shows the evaluation results of disease prediction for each of the configurations. For the two-sided Dunnett test, Levene's test shows that the equal variance condition is satisfied, and the Shapiro-Wilk test suggests normal distributions except for HRRAdd accuracy. The test showed HRRBase embeddings had a significantly greater mean accuracy $(p = 0.033)$ and precision $(p=0.023)$ compared to unstructured embeddings. No other comparisons of mean metrics for HRR embeddings were significantly greater than the control. 

\subsubsection{eICU Mortality Prediction}
An additional experiment conducted on the Philips Electronic Intensive Care Unit (eICU) \cite{pollard2018eicu} shows corroborating results with the MIMIC-IV experiments. 
% More details of the eICU experiments are reported in Appendix \ref{app:eicu}. 
For our experiment, we applied our mortality prediction models that were fine-tuned on MIMIC-IV to eICU data to see if our results generalize. \autoref{table:finetuneprediction} shows that HRRBase embeddings had a significantly greater mean accuracy ($p=0.046$) compared to unstructured embeddings when applied to the eICU dataset. These models are not optimized for mortality prediction for other hospitals where coding methodology and clinical practice may differ. For example, the most common code in the eICU dataset represents acute respiratory failure, whereas the most common code in the MIMIC-IV dataset represents hypertension.

\subsubsection{Really-Out-Of-Distribution (ROOD) Disease Prediction}
We conducted an additional disease-prediction experiment to test generalization to patients with codes outside the training distribution. We found six patients with records that consisted of only 32 codes between them (see list of codes in Appendix \ref{app:rood}). We created a really-out-of-distribution (ROOD) dataset that consisted of all patients in MIMIC-IV (nearly 30K) with at least one of these codes. We used this as a validation set. The separate pre-training and fine-tuning dataset did not contain these codes. We also created a smaller validation dataset consisting of the six patients with only these codes. 
% These 6 patients contain records spanning 32 total visits that only contain codes from the 32 selected ICD codes. 
% We create pre-training and fine-tuning datasets that contain records in MIMIC-IV that have no ROOD codes.
During pretraining, the HRRBase and unstructured models did not encounter any examples using the 32 ROOD codes and so did not explicitly learn representations for those codes. The trained models were then tested using the ROOD dataset.

Results from \autoref{table:finetuneprediction} on ROOD dataset disease prediction show that HRRBase outperforms the unstructured embedding model for contexts of entirely unseen codes. We assess statistical significance using two-tailed, independent t-test with unequal variance, as some measurements failed  Levene's test for equal variance. The means of all the metrics for HRRBase are significantly greater than for unstructured when making inferences on patients with entirely unseen codes, $p<0.001$ for all metrics. Given the embedded ontological structure, we hypothesize that HRRBase implicitly learns useful embeddings for the 32 unseen ROOD codes by learning any shared embedding components of the VSA when training on other codes. Unstructured embeddings cannot learn better representations for codes never seen in training. 

\subsection{t-SNE of Frequency Bias}
\begin{figure}[h]
\begin{center} 
\includegraphics[width=1\linewidth]{figures/4graphs.png} 
% \fbox{\rule[-.5cm]{0cm}{4cm} \rule[-.5cm]{4cm}{0cm}}
\end{center}
\caption{Comparing t-SNE of (a) unstructured embeddings, (b) HRRAdd, (c) HRRCat, and (d) HRRBase. The t-SNE graphs are color-coded by the frequency of the ICD codes in the dataset - highly frequent codes are colored blue while infrequent codes are colored red.}\label{fig:4graph}
\end{figure}

We computed t-SNE dimension reductions to visualize relationships among ICD code embeddings in the pre-trained models. \autoref{fig:4graph} shows that unstructured embeddings of common ICD codes are clustered together with a large separation from those of uncommon codes. This suggests that code-frequency information is prominently represented in these embeddings, consistent with frequency bias in related models \cite{freqbias}. 
Common and uncommon code clusters are less distinct in HRRBase, which does not explicitly encode frequency information. 

\begin{figure}[h]
\begin{center} 
\includegraphics[width=1\linewidth]{figures/3_cls_sim_rand_codes.png} 
% \fbox{\rule[-.5cm]{0cm}{4cm} \rule[-.5cm]{4cm}{0cm}}
\end{center}
\caption{t-SNE representation of sinusoidal frequency embeddings (left), and unstructured embedding components of HRRAdd (middle) and HRRCat (right).}\label{fig:fully}
\end{figure}

As shown in Figure \ref{fig:pretraining}, adding code-frequency information to the structured HRRBase embeddings, i.e. the HRRFreq embeddings, improved the pre-training loss be similar to unstructured embeddings. This suggests that unstructured components in HRRAdd and HRRCat may have learned some frequency information, since these losses are also similar to the loss of models with Unstructured embeddings. To investigate whether this occurred, we performed t-SNE dimension reductions of the unstructured components of HRRAdd and HRRCat and colored the points by code frequency, shown in Figure \ref{fig:fully}. This graph suggests that these additional unstructured embeddings learn some frequency information, due to clustering of high frequency codes. However, the frequency information learned by HRRCat and HRRAdd learnable embeddings influence overall embeddings less strongly in comparison to unstructured embeddings as seen in \autoref{fig:4graph}, where low frequency embeddings are less distinctly separated from higher frequency embeddings. 

\subsection{Top-k Accuracy for MLM}

Accurately predicting infrequently used disease codes is an important clinically relevant task. Given that the model trains and sees more common codes compared to rare codes, rare codes are naturally challenging to predict. Through promising empirical results on out-of-distribution mortality prediction for eICU and disease prediction on ROOD, we hypothesized that our HRR embedding models should have improved accuracy when predicting rare codes in the dataset compared to unstructured embedding models, since rare codes should share some atomic vectors in their representations with common codes. 

To test this, we evaluated the accuracy of an MLM pre-trained model predicting a single masked code of a known frequency. We split the codes in the pre-training validation dataset into 7 bins from log frequency -14 to 0, such that each bin has a width of 2. The most common codes are in a bin with log frequencies between -2 and 0, while the rarest codes are from a bin with log frequencies between -14 and -12. From each bin, we selected 400 codes at random, repeating codes from that bin if there were fewer than 400. For each of these codes, we selected one patient that had that code in their history, masked that code as would be done in MLM, and created a dataset of these 2,800 patients to use for MLM inference.

\begin{figure}
\begin{center} 
\includegraphics[width=1\linewidth]{figures/mlm-top-10-bars.png} 
% \fbox{\rule[-.5cm]{0cm}{4cm} \rule[-.5cm]{4cm}{0cm}}
\end{center}
\caption{The top-10 MLM accuracy for binned code frequencies in log scale. Common codes are in frequency bin 0 with rarest codes being in frequency bin -12. 0.05, 0.01, and 0.001 significance levels comparing to unstructured embeddings are indicated with 1, 2, and 3 asterisks respectively. Note that HRRBase is expected to perform poorly in this test due to lack of code-frequency information. }\label{fig:top10bars}
\end{figure}

\begin{figure}
\begin{center} 
\includegraphics[width=1\linewidth]{figures/mlm-top-100-bars.png} 
% \fbox{\rule[-.5cm]{0cm}{4cm} \rule[-.5cm]{4cm}{0cm}}
\end{center}
\caption{The top-100 MLM accuracy for binned code frequencies in log scale. Common codes are in frequency bin 0 with rarest codes being in frequency bin -12. 0.05, 0.01, and 0.001 significance levels comparing to unstructured embeddings are indicated with 1, 2, and 3 asterisks respectively.}\label{fig:top100bars}
\end{figure}

\autoref{fig:top10bars} and \autoref{fig:top100bars}, respectively, show the MLM top-10 and Top-100 accuracy on predicting codes in the different frequency bins, averaged across the three pre-training models per configuration. Significant comparisons to the unstructured control at a $p < 0.05$ level indicated with an asterisk. We assess statistical significance for each bin using a two-tailed Dunnett's test comparing mean accuracy scores of experimental HRR configurations against the control unstructured configuration.
Notably, the top-100 accuracy in frequency bin -12 is non-zero for the HRR methods. These codes in the rarest bin occur only once in the dataset and therefore have never been used by the model for gradient updates, since they are in the validation dataset. This suggests that the HRR methods have some ability to provide clinically relevant information about rare codes. However, accuracy with the rarest codes remains too low to be of practical value, perhaps due to limited overlap of these codes' atomic vectors with those of more common codes.
% , though only when given the ability to consider more than one possible label. 

\subsection{Medical Code Case Study}

\begin{table*}[t]
\caption{Three cosine similarity case studies looking at related ICD codes for unstructured and HRRBase. The top 4 cosine-similar ICD codes to the chosen code are listed (most to least similar) with their full description and similarity value.}\label{table:verify}
\label{table:example}
\begin{center}
\begin{tabular}{cccc}
    \hline
    \multicolumn{4}{c}{2724-9 - Other and unspecified hyperlipidemia}\\
    \hline
    \multicolumn{2}{c}{Unstructured} & \multicolumn{2}{c}{HRRBase}\\
    \hline
    Pure hypercholesterolemia&0.542&Other hyperlipidemia&1.000\\
    Hyperlipidemia, unspecified&0.482&Hyperlipidemia, unspecified&1.000\\
    Esophageal reflux&0.304&Pure hypercholesterolemia&0.463\\
    Anemia, unspecified&0.279&Mixed hyperlipidemia&0.418\\
    \hline
    % \hline
    \multicolumn{4}{c}{9916-9 - Hypothermia}\\
    \hline
    \multicolumn{2}{c}{Unstructured} & \multicolumn{2}{c}{HRRBase}\\
    \hline
    Frostbite of hand&0.418&Hypothermia, initial encounter&0.794\\
    Frostbite of foot&0.361&Hypothermia not with low env. temp.&0.592\\
    Drowning and nonfatal submersion&0.352&Effect of reduced temp., initial encounter&0.590\\
    Immersion foot&0.341&Other specified effects of reduced temp.&0.590\\
    \hline
    % \hline
    \multicolumn{4}{c}{K219-10 - Gastro-esophageal reflux disease without esophagitis}\\
    \hline
    \multicolumn{2}{c}{Unstructured} & \multicolumn{2}{c}{HRRBase}\\
    \hline
    Esophageal reflux&0.565&Esophageal reflux&0.635\\
    Hyperlipidemia, unspecified&0.335&Gastro-eso. reflux d. with esophagitis&0.512\\
    Anxiety disorder, unspecified&0.332&Reflux esophagitis&0.512\\
    Essential (primary) hypertension&0.326&Hypothyroidism, unspecified&0.268\\
    \hline
\end{tabular}
\end{center}
\end{table*}


\autoref{table:example} shows case studies for codes \textit{Other and unspecified hyperlipidemia} (2724-9), \textit{Hypothermia} (9916-9), and \textit{Gastro-esophageal Reflux disease without esophagitis} (K219-10). In the first case study for 2724-9, we observe highly ontologically similar codes, such as \textit{Other hyperlipidemia} and \textit{Hyperlipidemia, unspecified}, are encoded with high cosine similarity for HRRBase, which is not the case for unstructured embeddings. The co-occurrence problem can be seen in the second case study for 9916-9. The most similar codes for HRRBase are medically similar codes that would not usually co-occur, while for unstructured embeddings the most similar codes co-occur frequently. For the final case study on K219-10, frequency-related bias can be observed in the unstructured embeddings with frequent but mostly ontologically unrelated codes as part of the top list of cosine similar codes, whereas the top list of cosine similar codes for HRRBase contains medically similar codes.

We broadened this case study to test statistical differences in cosine and semantic embedding similarity between structured and unstructured embeddings. 30 ICD codes were selected from different frequency categories in the dataset, with 10 codes drawn randomly from the 300  most common codes, 10 codes drawn randomly by weighted frequency from codes appearing fewer than 30 times in the dataset, and 10 codes randomly selected by weighted frequency from the entire dataset. For each selected code, the top 4 cosine-similar ICD codes were assessed by a physician for ontological similarity. 

For each frequency category, a one-tailed Fisher's exact test was conducted 
%with a significance level of $p < 0.05$ 
to determine whether a relationship existed between embedding type and clinical relatedness. 
We found that results in the case of the rare codes were statistically significant, with $p = 2.44 \times 10^{-8}$. With 10 rare codes and the top 4 cosine-similar ICD codes selected for each rare code, there are 40 top cosine-similar codes in total. In the case of unstructured embeddings, only 4 of the top 40 cosine-similar codes were deemed to be strongly ontologically related by our physician with the remaining codes deemed to be less related and unrelated. In the case of our structured HRRBase embeddings, 28 of the top 40 cosine-similar codes were deemed to be strongly ontologically related by our physician with the remaining codes deemed to be less related and unrelated. 
% Knowledge-integrated learning seems to improve embedding quality in terms of clinical relevance for rare codes where little training data exists.  
This suggests that knowledge-integrated structured embeddings are associated with greater clinical relevance of the top cosine-similar codes than unstructured embeddings for rare codes where little training data exists. 
% Using a one-tailed Fisher's exact test, we found a statistically significant relationship between embedding type and clinical relatedness for rare codes, with $p = 2.44 \times 10^{-8}$. This suggests that structured embeddings are associated with greater clinical relevance of the top cosine-similar codes than unstructured embeddings. 

\section{Discussion}

% This study shows how structured knowledge from a medical ontology can be incorporated into the learned embeddings for an LLM to efficiently reduce learned biases and improve performance. 
% Though standard unstructured embeddings perform well, their concept representations have reduced grounding in medical ontology. 
% When compared to the unstructured approach, our novel neuro-symbolic HRRBERT approach better reflects ontological meaning and improves performance in the fine-tuning tasks of mortality and disease prediction, especially in cases containing rare and infrequent codes.
% Rare and infrequent medical codes and conditions pose a challenge for both clinicians and medical LLMs alike; our results show that our proposed neuro-symbolic LLM architecture HRRBERT better serves as a tool to clinicians by improving performance in cases containing rare and infrequent codes. 
% Comparing learned embeddings between HRRBERT and unstructured models, HRRBERT embeddings show less learned frequency bias and more so reflect medical knowledge in terms of embeddings for medical synonyms, and solving the co-occurrence problem for learned medical embeddings. 

Transformers have leading performance in many applications, but their internal processes are opaque, emerging from enormous parameter sets and data volumes beyond human experience. It is hard to know when they can be trusted. For example, generative transformers are prone to subtle confabulations. Transformers have a general-purpose architecture that performs as well in vision and other modalities as in language. They are a culmination of a key trend in artificial intelligence, away from problem-specific engineering, and toward massive data and computation. This trend is justified in terms of performance. However, given two models with equal performance, one with more explicit conceptual structure is preferable in terms of trust and explainability.

The work presented here is a step in this direction, with our HRRBase embeddings that have explicit conceptual structure and perform equivalently or better compared to typical transformer embeddings. The benefit of structured embeddings becomes more pronounced for tasks that involve codes that are rare or are not present in training data. HRR embeddings can also be relied on to represent medical meaning rather than co-occurrence in the training data. They also untangle the representation of code frequency, so that it can be included or not, and its effects on decisions understood. Importantly, despite this additional structure, the embeddings are thoroughly learned, suggesting that the approach will be consistent with high performance beyond the examples we have studied. 

As our method scales with and leverages PyTorch autograd in the construction of the vector-symbolic embeddings, it is compatible with existing medical LLM architectures as an embedding component capable of encoding domain knowledge.

Future work could explore the potential of these structured embeddings for explaining and controlling the observed frequency bias.
As HRRs can be queried with linear operations, future work could also explore whether transformers can learn to extract specific information from these composite embeddings. 
Limitations to address in future work include the complexity of processing knowledge graphs to be compatible with HRRs.
Another important limitation is that our method relies on rare-code HRRs sharing atomic elements with common-code HRRs. 
However, in SNOMED CT, rare codes are likely to contain some rare atomic elements.
To address this point, in addition to SNOMED CT, knowledge could be encoded from sources such as pre-trained medical embeddings, different medical ontologies, and other medical domain knowledge to further improve our proposed methodology. In LLMs that process both medical codes and text, it would make sense to share word embeddings between modalities. This would allow training of each modality to benefit from training of the other, and may help to align the representations of codes and text. 

\section{Conclusion}

% Note hyphen added to HRRBERT to fix where formatting error
We proposed a novel hybrid neural-symbolic approach called HRR-BERT that integrates medical ontologies represented by HRR embeddings. In tests with the MIMIC-IV dataset, HRRBERT models modestly outperformed baseline models with unstructured embeddings for pre-training, disease prediction accuracy, mortality prediction F1, and fine-tuning tasks involving infrequently seen codes. 
HRRBERT models had pronounced performance advantages in MLM with rare codes and disease prediction for patients with no codes seen during training (ROOD - Unseen in Table \ref{table:finetuneprediction}). 
We also showed that HRRs can be used to create medical code embeddings that better respect ontological similarities for rare codes. 
A key benefit of our approach is that it facilitates explainability by disentangling token-frequency information, which is prominently represented but implicit in unstructured embeddings. 
% Critical to this approach is a new method to construct vector-symbolic embeddings that leverage PyTorch autograd on GPUs, allowing learning through HRR operations. 

% \section{Modifications}

% Modifying the template --- including but not limited to: adjusting
% margins, typeface sizes, line spacing, paragraph and list definitions,
% and the use of the \verb|\vspace| command to manually adjust the
% vertical spacing between elements of your work --- is not allowed.

% \section{Template parameters}

% There are a number of template
% parameters which modify some part of the \verb|ceurart| document class.
% This parameters are enclosed in square
% brackets and are a part of the \verb|\documentclass| command:
% \begin{lstlisting}
%   \documentclass[parameter]{ceurart}
% \end{lstlisting}

% Frequently-used parameters, or combinations of parameters, include:
% \begin{itemize}
% \item \verb|twocolumn| : Two column layout.
% \item \verb|hf| : Enable header and footer\footnote{You can enable
%     the display of page numbers in the final version of the entire
%     collection. In this case, you should adhere to the end-to-end
%     pagination of individual papers.}.
% \end{itemize}

% \section{Front matter}

% \subsection{Title Information}

% The titles of papers should be either all use the emphasizing
% capitalized style or they should all use the regular English (or
% native language) style. It does not make a good impression if you or
% your authors mix the styles.

% Use the \verb|\title| command to define the title of your work. Do not
% insert line breaks in your title.

% \subsection{Title variants}

% \verb|\title| command have the below options:
% \begin{itemize}
% \item \verb|title|: Document title. This is default option. 
% \begin{lstlisting}
% \title[mode=title]{This is a title}
% \end{lstlisting}
% You can just omit it, like as follows:
% \begin{lstlisting}
% \title{This is a title}
% \end{lstlisting}

% \item \verb|alt|: Alternate title.
% \begin{lstlisting}
% \title[mode=alt]{This is a alternate title}
% \end{lstlisting}

% \item \verb|sub|: Sub title.
% \begin{lstlisting}
% \title[mode=sub]{This is a sub title}
% \end{lstlisting}
% You can just use \verb|\subtitle| command, as follows:
% \begin{lstlisting}
% \subtitle{This is a sub title}
% \end{lstlisting}

% \item \verb|trans|: Translated title.
% \begin{lstlisting}
% \title[mode=trans]{This is a translated title}
% \end{lstlisting}

% \item \verb|transsub|: Translated sub title.
% \begin{lstlisting}
% \title[mode=transsub]{This is a translated sub title}
% \end{lstlisting}
% \end{itemize}

% \subsection{Authors and Affiliations}

% Each author must be defined separately for accurate metadata
% identification. Multiple authors may share one affiliation. Authors'
% names should not be abbreviated; use full first names wherever
% possible. Include authors' e-mail addresses whenever possible.

% \verb|\author| command have the below options: 

% \begin{itemize}
% \item \verb|style| : Style of author name (chinese)
% \item \verb|prefix| : Prefix
% \item \verb|suffix| : Suffix
% \item \verb|degree| : Degree
% \item \verb|role| : Role
% \item \verb|orcid| : ORCID
% \item \verb|email| : E-mail
% \item \verb|url| : URL
% \end{itemize}

% Author names can have some kinds of marks and notes:
% \begin{itemize}
% \item affiliation mark: \verb|\author[<num>]|.
% \end{itemize}

% The author names and affiliations could be formatted in two ways:
% \begin{enumerate}
% \item Group the authors per affiliation.
% \item Use an explicit mark to indicate the affiliations.
% \end{enumerate}

% Author block example:
% \begin{lstlisting}
% \author[1,2]{Author Name}[%
%     prefix=Prof.,
%     degree=D.Sc.,
%     role=Researcher,
%     orcid=0000-0000-000-0000,
%     email=name@example.com,
%     url=https://name.example.com
% ]

% \address[1]{Affiliation #1}
% \address[2]{Affiliation #2}
% \end{lstlisting}

% \subsection{Abstract and Keywords}

% Abstract shall be entered in an environment that starts
% with \verb|\begin{abstract}| and ends with
% \verb|\end{abstract}|. 

% \begin{lstlisting}
% \begin{abstract}
%   This is an abstract.
% \end{abstract}
% \end{lstlisting}

% The key words are enclosed in a \verb|keywords|
% environment. Use \verb|\sep| to separate keywords.

% \begin{lstlisting}
% \begin{keywords}
%   First keyword \sep 
%   Second keyword \sep 
%   Third keyword \sep 
%   Fourth keyword
% \end{keywords}
% \end{lstlisting}

% At the end of front matter add \verb|\maketitle| command.

% \subsection{Various Marks in the Front Matter}

% The front matter becomes complicated due to various kinds
% of notes and marks to the title and author names. Marks in
% the title will be denoted by a star ($\star$) mark;
% footnotes are denoted by super scripted Arabic numerals,
% corresponding author by an Conformal asterisk (*) mark.

% \subsubsection{Title marks}

% Title mark can be entered by the command, \verb|\tnotemark[<num>]|
% and the corresponding text can be entered with the command
% \verb|\tnotetext[<num>]{<text>}|. An example will be:

% \begin{lstlisting}
% \title{A better way to format your document for CEUR-WS}

% \tnotemark[1]
% \tnotetext[1]{You can use this document as the template for preparing your
%   publication. We recommend using the latest version of the ceurart style.}
% \end{lstlisting}

% \verb|\tnotemark| and \verb|\tnotetext| can be anywhere in
% the front matter, but should be before \verb|\maketitle| command.

% \subsubsection{Author marks}

% Author names can have some kinds of marks and notes:
% \begin{itemize}
% \item footnote mark : \verb|\fnmark[<num>]|
% \item footnote text : \verb|\fntext[<num>]{<text>}|
% \item corresponding author mark : \verb|\cormark[<num>]|
% \item corresponding author text : \verb|\cortext[<num>]{<text>}|
% \end{itemize}

% \subsubsection{Other marks}

% At times, authors want footnotes which leave no marks in
% the author names. The note text shall be listed as part of
% the front matter notes. Class files provides
% \verb|\nonumnote| for this purpose. The usage
% \begin{lstlisting}
% \nonumnote{<text>}
% \end{lstlisting}
% and should be entered anywhere before the \verb|\maketitle|
% command for this to take effect. 

% \section{Sectioning Commands}

% Your work should use standard \LaTeX{} sectioning commands:
% \verb|\section|, \verb|\subsection|,
% \verb|\subsubsection|, and
% \verb|\paragraph|. They should be numbered; do not remove
% the numbering from the commands.

% Simulating a sectioning command by setting the first word or words of
% a paragraph in boldface or italicized text is not allowed.

% \section{Tables}

% The ``\verb|ceurart|'' document class includes the ``\verb|booktabs|''
% package --- \url{https://ctan.org/pkg/booktabs} --- for preparing
% high-quality tables.

% Table captions are placed \textit{above} the table.

% Because tables cannot be split across pages, the best placement for
% them is typically the top of the page nearest their initial cite.  To
% ensure this proper ``floating'' placement of tables, use the
% environment \verb|table| to enclose the table's contents and the
% table caption. The contents of the table itself must go in the
% \verb|tabular| environment, to be aligned properly in rows and
% columns, with the desired horizontal and vertical rules.

% Immediately following this sentence is the point at which
% Table~\ref{tab:freq} is included in the input file; compare the
% placement of the table here with the table in the printed output of
% this document.

% \begin{table*}
%   \caption{Frequency of Special Characters}
%   \label{tab:freq}
%   \begin{tabular}{ccl}
%     \toprule
%     Non-English or Math&Frequency&Comments\\
%     \midrule
%     \O & 1 in 1,000& For Swedish names\\
%     $\pi$ & 1 in 5& Common in math\\
%     \$ & 4 in 5 & Used in business\\
%     $\Psi^2_1$ & 1 in 40,000& Unexplained usage\\
%   \bottomrule
% \end{tabular}
% \end{table*}

% To set a wider table, which takes up the whole width of the page's
% live area, use the environment \verb|table*| to enclose the table's
% contents and the table caption.  As with a single-column table, this
% wide table will ``float'' to a location deemed more
% desirable. Immediately following this sentence is the point at which
% Table~\ref{tab:commands} is included in the input file; again, it is
% instructive to compare the placement of the table here with the table
% in the printed output of this document.

% \begin{table}
%   \caption{Some Typical Commands}
%   \label{tab:commands}
%   \begin{tabular}{ccl}
%     \toprule
%     Command &A Number & Comments\\
%     \midrule
%     \texttt{{\char'134}author} & 100& Author \\
%     \texttt{{\char'134}table}& 300 & For tables\\
%     \texttt{{\char'134}table*}& 400& For wider tables\\
%     \bottomrule
%   \end{tabular}
% \end{table}

% \section{Math Equations}

% You may want to display math equations in three distinct styles:
% inline, numbered or non-numbered display.  Each of the three are
% discussed in the next sections.

% \subsection{Inline (In-text) Equations}

% A formula that appears in the running text is called an inline or
% in-text formula.  It is produced by the \verb|math| environment,
% which can be invoked with the usual
% \verb|\begin| \ldots \verb|\end| construction or with
% the short form \verb|$| \ldots \verb|$|. You can use any of the symbols
% and structures, from $\alpha$ to $\omega$, available in
% \LaTeX~\cite{Lamport:LaTeX};
% this section will simply show a few
% examples of in-text equations in context. Notice how this equation:
% \begin{math}
%   \lim_{n\rightarrow \infty} \frac{1}{n} = 0,
% \end{math}
% set here in in-line math style, looks slightly different when
% set in display style.  (See next section).

% \subsection{Display Equations}

% A numbered display equation---one set off by vertical space from the
% text and centered horizontally---is produced by the \verb|equation|
% environment. An unnumbered display equation is produced by the
% \verb|displaymath| environment.

% Again, in either environment, you can use any of the symbols and
% structures available in \LaTeX{}; this section will just give a couple
% of examples of display equations in context.  First, consider the
% equation, shown as an inline equation above:
% \begin{equation}
%   \lim_{n\rightarrow \infty} \frac{1}{n} = 0.
% \end{equation}
% Notice how it is formatted somewhat differently in
% the \verb|displaymath|
% environment.  Now, we'll enter an unnumbered equation:
% \begin{displaymath}
%   S_{n} = \sum_{i=1}^{n} x_{i} ,
% \end{displaymath}
% and follow it with another numbered equation:
% \begin{equation}
%   \lim_{x \to 0} (1 + x)^{1/x} = e
% \end{equation}
% just to demonstrate \LaTeX's able handling of numbering.

% \section{Figures}

% The ``\verb|figure|'' environment should be used for figures. One or
% more images can be placed within a figure. If your figure contains
% third-party material, you must clearly identify it as such, as shown
% in the example below.
% \begin{figure}
%   \centering
%   \includegraphics[width=\linewidth]{sample-franklin}
%   \caption{1907 Franklin Model D roadster. Photograph by Harris \&
%     Ewing, Inc. [Public domain], via Wikimedia
%     Commons. (\url{https://goo.gl/VLCRBB}).}
% \end{figure}

% Your figures should contain a caption which describes the figure to
% the reader. Figure captions go below the figure. Your figures should
% also include a description suitable for screen readers, to
% assist the visually-challenged to better understand your work.

% Figure captions are placed below the figure.

% \section{Citations and Bibliographies}

% The use of Bib\TeX{} for the preparation and formatting of one's
% references is strongly recommended. Authors' names should be complete
% --- use full first names (``Donald E. Knuth'') not initials
% (``D. E. Knuth'') --- and the salient identifying features of a
% reference should be included: title, year, volume, number, pages,
% article DOI, etc.

% The bibliography is included in your source document with these two
% commands, placed just before the \verb|\end{document}|
% command:
% \begin{lstlisting}
% \bibliography{bibfile}
% \end{lstlisting}
% where ``\verb|bibfile|'' is the name, without the ``\verb|.bib|''
% suffix, of the Bib\TeX{} file.


% \subsection{Some examples}

% A paginated journal article \cite{Abril07}, an enumerated journal
% article \cite{Cohen07}, a reference to an entire issue
% \cite{JCohen96}, a monograph (whole book) \cite{Kosiur01}, a
% monograph/whole book in a series (see 2a in spec. document)
% \cite{Harel79}, a divisible-book such as an anthology or compilation
% \cite{Editor00} followed by the same example, however we only output
% the series if the volume number is given \cite{Editor00a} (so series
% should not be present since it has no vol. no.), a chapter in a
% divisible book \cite{Spector90}, a chapter in a divisible book in a
% series \cite{Douglass98}, a multi-volume work as book \cite{Knuth97},
% an article in a proceedings (of a conference, symposium, workshop for
% example) (paginated proceedings article) \cite{Andler79}, a
% proceedings article with all possible elements \cite{Smith10}, an
% example of an enumerated proceedings article \cite{VanGundy07}, an
% informally published work \cite{Harel78}, a doctoral dissertation
% \cite{Clarkson85}, a master's thesis: \cite{anisi03}, an online
% document / world wide web resource \cite{Thornburg01, Ablamowicz07,
%   Poker06}, a video game (Case 1) \cite{Obama08} and (Case 2)
% \cite{Novak03} and \cite{Lee05} and (Case 3) a patent
% \cite{JoeScientist001}, work accepted for publication \cite{rous08},
% prolific author \cite{SaeediMEJ10} and \cite{SaeediJETC10}. Other
% cites might contain `duplicate' DOI and URLs (some SIAM articles)
% \cite{Kirschmer:2010:AEI:1958016.1958018}. Multi-volume works as books
% \cite{MR781536} and \cite{MR781537}. A couple of citations with DOIs:
% \cite{2004:ITE:1009386.1010128,Kirschmer:2010:AEI:1958016.1958018}. Online
% citations: \cite{TUGInstmem, Thornburg01, R, UMassCitations}.

% \section{Acknowledgments}

% Identification of funding sources and other support, and thanks to
% individuals and groups that assisted in the research and the
% preparation of the work should be included in an acknowledgment
% section, which is placed just before the reference section in your
% document.

% This section has a special environment:
% \begin{lstlisting}
% \begin{acknowledgments}
%   These are different acknowledgments.
% \end{acknowledgments}
% \end{lstlisting}
% so that the information contained therein can be more easily collected
% during the article metadata extraction phase, and to ensure
% consistency in the spelling of the section heading.

% Authors should not prepare this section as a numbered or unnumbered
% \verb|\section|; please use the ``\verb|acknowledgments|'' environment.

% \section{Appendices}

% If your work needs an appendix, add it before the
% ``\verb|\end{document}|'' command at the conclusion of your source
% document.

% Start the appendix with the ``\verb|\appendix|'' command:
% \begin{lstlisting}
% \appendix
% \end{lstlisting}
% and note that in the appendix, sections are lettered, not
% numbered. 

%%
%% The acknowledgments section is defined using the "acknowledgments" environment
%% (and NOT an unnumbered section). This ensures the proper
%% identification of the section in the article metadata, and the
%% consistent spelling of the heading.
% \begin{acknowledgments}
%   Thanks to the developers of ACM consolidated LaTeX styles
%   \url{https://github.com/borisveytsman/acmart} and to the developers
%   of Elsevier updated \LaTeX{} templates
%   \url{https://www.ctan.org/tex-archive/macros/latex/contrib/els-cas-templates}.  
% \end{acknowledgments}

%%
%% Define the bibliography file to be used
\bibliography{sample-ceur}

%%
%% If your work has an appendix, this is the place to put it.
\appendix


\section{List of 32 ROOD Codes}\label{app:rood}
The following is the list of 32 ROOD codes:
\begin{enumerate}
  \item G248-10: Other dystonia
\item E8498-9: Accidents occurring in other specified places
\item E9688-9: Assault by other specified means
\item Z681-10: Body mass index (BMI) 19.9 or less, adult
\item 30550-9: Opioid abuse, unspecified
\item R262-10: Difficulty in walking, not elsewhere classified
\item E887-9: Fracture, cause unspecified
\item R471-10: Dysarthria and anarthria
\item 9916-9: Hypothermia
\item E9010-9: Accident due to excessive cold due to weather conditions
\item F10129-10: Alcohol abuse with intoxication, unspecified
\item E8499-9: Accidents occurring in unspecified place
\item R636-10: Underweight
\item 920-9: Contusion of face, scalp, and neck except eye(s)
\item R4182-10: Altered mental status, unspecified
\item 95901-9: Head injury, unspecified
\item 78097-9: Altered mental status
\item F29-10: Unspecified psychosis not due to a substance or known physiological condition
\item Z880-10: Allergy status to penicillin
\item Z818-10: Family history of other mental and behavioral disorders
\item 81600-9: Closed fracture of phalanx or phalanges of hand, unspecified
\item 87341-9: Open wound of cheek, without mention of complication
\item H9222-10: Otorrhagia, left ear
\item Z978-10: Presence of other specified devices
\item G20-10: Parkinson's disease
\item G249-10: Dystonia, unspecified
\item 9100-9: Abrasion or friction burn of face, neck, and scalp except eye, without mention of infection
\item 78906-9: Abdominal pain, epigastric
\item E8889-9: Unspecified fall
\item 30500-9: Alcohol abuse, unspecified
\item G520-10: Disorders of olfactory nerve
\item 8020-9: Closed fracture of nasal bones
\end{enumerate}

\subsection{Learning through HRR Operations Efficiently}\label{app:learn-hrr}
To make the HRR concept embeddings useful for a deep neural network, the operations used to form the embeddings need to be compatible with backpropagation so that gradient descent can update the lower-level atomic vectors. We desired a function that produced the ICD concept embedding matrix, $\mC$, given the inputs of the VSA knowledge graphs, $\mathcal{G}_i$, and symbol embedding matrices, $\mR$ and $\mA$. 

We attempted three approaches to computing $\mC$ through VSA operations. First, we naively tried to compute each concept vector in $\mC$ one at a time. However, this approach was too slow in both forward and backward pass, requiring more than 1 second for each pass. Our second approach was using slices of $\tG$ along the relationship dimension as a sparse binary matrix, which, when multiplied with $\mA$, would perform the indexing and summing of atomic vectors for each concept. This result can be convolved with the relationship vector and added to the concept embedding matrix. This approach was much faster and used a moderate amount of memory for one of our less complex VSA formulations. However, when dealing with our most complex formulation, it used $\sim$15 GB of memory.
% which reduced the remaining RAM of the model by 55\% on our RTX 3090s and would make this algorithm unusable on smaller GPUs. 

Our final approach took advantage of the fact that many disease concepts use relationship, but to different atomic symbols. Also, number of times a concept uses a particular relationship is relatively low, except for the SNOMED ``isA'' relationship and our defined ``description'' relationship. Thus, for a particular relationship, we can contribute to building many disease concept vectors at once by selecting many atomic vectors, doing a vectorized convolution with the relationship vector, and distributing the results to be added with the appropriate concept embedding rows. This step needs to be repeated at most $m$ times for a particular relationship, where $m$ is the maximum multiplicity of that relationship among all concepts. We improved memory efficiency by performing fast Fourier transforms (FFTs) on the atomic vector embeddings and construct the concept vectors by performing binding via element-wise multiplication in the Fourier domain. Due to the linearity of the HRR operations, we performed a final FFT on the complex-valued concept embedding to convert back to the real domain. 
% This saved both time and memory since each circular convolution would have needed to perform two FFT operations, but we combined those all at the beginning and end of the algorithm. 

The final approach is much faster than the first approach since it takes advantage of vectorized operations to contribute to many concept vectors at once. 
It is also more memory efficient than the second approach since all the intermediate results are dense, so allocations are not wasted on creating mostly sparse results. 
On our most complex formulation, this approach uses $\sim$3.5 GB of memory, and takes $\sim$80 ms and $\sim$550 ms for forward and backward pass respectively.

\end{document}

%%
%% End of file
