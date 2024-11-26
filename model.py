import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


class BertForWordBoundaryDetection(nn.Module):
    def __init__(
        self,
        model_name="lassl/bert-ko-small",
        digit_token="<N>",
        max_length=6
    ):
        super(BertForWordBoundaryDetection, self).__init__()

        # Load pretrained BERT model and tokenizer
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add special token and resize the embeddings
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [digit_token]}
        )
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Linear layer for classification
        self.classifier = nn.Linear(self.model.config.hidden_size, 1)
        
        # Max length for tokenization
        self.max_length = max_length

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        # Pass input through BERT model
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extract the CLS token's representation
        cls_output = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
        
        # Pass CLS output through the classifier
        logits = self.classifier(cls_output)  # (batch_size, 1)
        return logits

    def tokenize_function(self, texts: str | list[str]):
        # Tokenize inputs with padding, truncation, and maximum length
        return self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
