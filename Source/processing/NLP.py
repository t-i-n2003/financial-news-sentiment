from transformers import AutoTokenizer, AutoModel
import torch
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# model = "vinai/phobert-base"
# phobert = AutoModel.from_pretrained(model)
# tokenizer = AutoTokenizer.from_pretrained(model)
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# def get_embedding(text):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
#     with torch.no_grad():
#         outputs = phobert(**inputs)
#     embeddings = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
#     pooled = embeddings.mean(dim=1)  # [batch_size, hidden_size]
#     return pooled.squeeze().numpy()

def natural_language_process(text, symbols):   
    pattern = r'\b(?:' + '|'.join(symbols) + r')\b'
    text = re.sub(pattern, '', text).strip()
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = word_tokenize(text)
    processed_tokens = [word for word in tokens if word not in stop_words]

    return " ".join(processed_tokens)
