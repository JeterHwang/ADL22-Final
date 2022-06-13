import json
import pickle
from typing import List, Dict
import torch
from torch import cuda
from torch.nn import Embedding, Linear, LSTM
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils import Vocab
import random

device = 'cuda' if cuda.is_available() else 'cpu'
HIDDEN_SIZE = 512
NUM_LAYERS = 2
DROP_OUT = 0.1
BIDIRECTIONAL = True

class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.hidden_size = hidden_size
        self.lstm = LSTM(input_size=300, hidden_size=self.hidden_size, num_layers=num_layers,
                          batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.linear = Linear((int(bidirectional) + 1) * hidden_size, num_class)

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        embeded = self.embed(batch)
        batch_len = len(batch[0])
        output, (h_n, c_n) = self.lstm(embeded)
        y = self.linear(output[:,batch_len - 1,:])
        return y

def predict_keyword_lstm(input: List[str], vocab_path: str, embedding_path: str, model_path: str, subdomain_path:str) -> str:
    with open(vocab_path, "rb") as f:
        vocab : Vocab = pickle.load(f)
    input = ' '.join(input)
    word_list = input.split()
    encoded_input = vocab.encode(word_list)

    with open(subdomain_path, 'r') as f:
        raw_domain = json.load(f)

    subdomain_list = list(raw_domain.keys())
  
    embeddings = torch.load(embedding_path)

    model = SeqClassifier(embeddings=embeddings, 
                          hidden_size= HIDDEN_SIZE, 
                          num_layers= NUM_LAYERS,
                          dropout= DROP_OUT,
                          bidirectional= BIDIRECTIONAL,
                          num_class=len(subdomain_list)).to(device)
    model.eval()
    model.load_state_dict(torch.load(model_path))

    model_input = torch.Tensor([encoded_input]).long().to(device)
    output = model(model_input)
    subdomain = torch.argmax(output)
    candidate = raw_domain[subdomain_list[subdomain]]

    rand_index = random.randint(0, len(candidate) - 1)
    return candidate[rand_index]