import torch
import torch.nn.functional as F
import transformers

# from transformers import AutoModel, AutoTokenizer


class FeatureExtractionPipeline(transformers.FeatureExtractionPipeline):
    def _sanitize_parameters(self, **kwargs):
        # Handle pipeline-specific arguments

        tokenize_kwargs, forward_kwargs, postprocess_kwargs = super()._sanitize_parameters(**kwargs)
        if 'prefix' in kwargs:
            tokenize_kwargs['prefix'] = kwargs['prefix'] # 'passage' or 'query'
        return tokenize_kwargs, forward_kwargs, postprocess_kwargs

    def preprocess(self, texts, prefix: str|None=None, **tokenize_kwargs):
        # Apply the required prefix: 'passage: ' for documents, 'query: ' for queries
        if prefix is not None:
            texts = [f"{prefix}: {text}" for text in texts]

        # Tokenize the batch
        return super().preprocess(texts, **tokenize_kwargs)


class E5EmbeddingPipeline(FeatureExtractionPipeline):

    def _forward(self, model_inputs):
        # Perform the model forward pass
        return self.model(**model_inputs, return_dict=True)

    def postprocess(self, model_outputs):
        # 1. Perform mean pooling using the attention mask
        token_embeddings = model_outputs.last_hidden_state
        attention_mask = model_outputs.attention_mask
        
        # Expand mask to match embedding dimensions [batch_size, seq_len, 1]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Sum embeddings, then divide by the sum of mask values
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeddings = sum_embeddings / sum_mask
        
        # 2. Normalize embeddings (CRITICAL for E5 models)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().tolist()

# Initialize the custom pipeline
# e5_embed_pipeline =  EmbeddingPipeline(
#     model=AutoModel.from_pretrained("intfloat/multilingual-e5-large"),
#     tokenizer=AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large"),
#     device=0 if torch.cuda.is_available() else -1,  # Use GPU if available
#     torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
# )
# )
# )
