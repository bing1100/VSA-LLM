Looking tokenization / embedding

Tokenization is generated through bpe will learn the best tokens to decompose text to - where then each token is given a learnable embedding 

A transformer model will them learn and update the semantic embeddings of the tokenization. 

In a way, the semantic information flows through learning transformer models to embeddings - but tokenization is not itself learnable - the structure of tokenization outlines an optimal syntax, not the optimal semantic construction. This syntax to semantic jump is bridged through large amounts of model parameters - where adding in additional training data and parameters allows the encoding of the syntax to semantic jump for large models. 

VSA-llm looks to bridge this gap. 

VSA looks to generate a semantic representation breakdown of concepts that occur often - they can be learned in parallel to syntax 

VSA encodes semantic ontologies for how meanings can be combined using relationships and atomic vectors summed together. In a way, this is similar to attention formulation in of itself - where we have (relationship_vector x mapping) circlular convolved with atomic vectors - summing together to form the concept itself. Somehow we should also work in the query vector in this formulation - such that we can incorporate in the context from the query to generate and modify the mapping - where a word definition and semantic meaning may differ based on that context. 

Through this formulation, the relationship and atomic vectors become learnable. But we also want to make the mapping itself learnable - for this we could use a svd for the mapping matrix - similar to lora - reducing to vector A x B - where A and B vectors can be learned.

We would also like to learn the number of relationships and be able to add in new atomic vectors as well. In order to do this, we need to track two values - the momentum of gradient for the vector itself - and the mean absolute value of the gradients - in the case where the momentum of the vector itself is close to 0, the the mean absolute value of the gradient is high, this means that gradients are pulling in different directions - thus this atomic vector or relationship vector need to be split into two different vectors - where at the begining the mapping and vector values are the same, but they are allowed to drift apart.

A possible application of VSA is as an additional embedding to add to the sequence - like position embeddings 

Semantic embeddings should not be created for every token - instead they should be created for words with multiple tokens - where words with multiple tokens are extremely hard to learn as the model needs to learn to bridge the syntax to semantic meaning in it's own weights. 

Why VSA helps:

- VSA can improve the computation efficiency of models - where with fewer parameters and less data, we're able to achieve the same performance
- since VSA improves model efficiency, it can best be used to improve the performance of quantized and smaller models especially on complex tasks and vocabulary
- VSA can also be used for explanability - where through mapping atomic vectors and defining ontologies, these embeddings can be explain how the model views the world
- through VSA we can do zero-shot learning - where by defining new ontologies through these mappings, we can give the model an idea of the semantic meaning 






