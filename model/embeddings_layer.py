from typing import Optional

import torch
from transformers import BertTokenizerFast, BertModel

from .bert_encoder import BertEncoder

tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained('bert-base-uncased')
model: BertModel = BertModel.from_pretrained("bert-base-uncased")
print()


class EmbeddingsLayer:
    def __init__(self, dense1, dense2, proj1, proj2, word2vec, ontology,
                 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()

        self.device = device
        self.tokenizer: BertTokenizerFast = tokenizer
        self.model: BertModel = model.to(device)
        self.encoder = BertEncoder(self.model, dense1=dense1, dense2=dense2, proj1=proj1, proj2=proj2, word2vec=word2vec, ontology=ontology)

    def forward(self, sentence: str, target_start: int, target_end: int, knowledge_layers):
        # Tokenize the sentence and get offsets
        encoding = self.tokenizer(
            sentence,
            return_offsets_mapping=True,
            add_special_tokens=True,
            return_tensors="pt"
        ).to(self.device)

        input_ids = encoding["input_ids"]             # shape: [1, seq_len]
        token_type_ids = encoding["token_type_ids"]   # shape: [1, seq_len]
        offsets = encoding["offset_mapping"][0]       # shape: [seq_len, 2]

        # Find token indices corresponding to character positions
        target_token_start, target_token_end = None, None
        for i, (start_char, end_char) in enumerate(offsets.tolist()):
            if start_char <= target_start < end_char:
                target_token_start = i
            if start_char < target_end <= end_char:
                target_token_end = i + 1  # inclusive
                break

        if target_token_start is None or target_token_end is None:
            raise ValueError(f"Cannot map character indices {target_start}-{target_end} to tokens.")

        # Get initial embeddings
        initial_embeddings = self.model.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)

        embeddings = self.encoder(initial_embeddings, sentence=self.tokenizer.convert_ids_to_tokens(input_ids[0]), knowledge_layers=knowledge_layers)

        # remove [CLS] and [SEP] if needed
        embeddings = embeddings[0][1:-1]

        return embeddings, (target_token_start-1, target_token_end-1), None