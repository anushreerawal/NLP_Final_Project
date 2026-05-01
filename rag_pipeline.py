
import json
import time
import urllib.request
import zipfile
import io
import numpy as np

from typing import List
from dataclasses import dataclass, field

# optional imports

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        "sentence-transformers is not installed. Run:\n"
        "pip install sentence-transformers"
    )

try:
    import faiss
except ImportError:
    raise ImportError(
        "faiss-cpu is not installed. Run:\n"
        "pip install faiss-cpu"
    )

'''
structures
'''

@dataclass
class LegalPassage:
    passage_id: str
    text: str
    source str
    case_name: str = ""
    date: str =  ""
    score: float = 0.0

@dataclass
class RAGResult:
    question: str
    retrieved_passages: List[LegalPassage]
    generated_answer: str
    retrieval_score: int
    answer_score: int
    latency_mas: float = 0.0

# Dataset loader for ContractNLI

class ContractNLILoader:
    URL = "https://stanfordnlp.github.io/contract-nli/resources/contract-nli.zip"

    def __init__(Self, max_passages: int = 500):
        self.max_passages = max_passages
    
    def load_corpus(self) -> List[LegalPassage]:
        print("downloading ContractNLI dataset...")
    
    with urllib.request.urlopen(self.URL, timeout=60) as response:
        data = response.read()
    
    passages = []

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = zf.namelist()
        json_files = [name for name in names if name.endswith(".json")]

        if not json_files:
            raise ValueError("No JSON file found in ContractNLI zip.")

        train_file = next(
            (name for name in json_files if "train" in name.lower()), 
            json_files[0]
        )

        with zf.open(train_file) as f:
            corpus = json.load(f)
        
    docs = self._extract_documents(corpus)

    if not docs:
        raise ValueError("Could not find documents inside ContractNLI data.")
    
    for doc_idx, dox in enumerate(docs):
        if len(passages) >= self.max_passages:
            break

        text = self._extract_text(doc)

        if not text.strip():
            continue

        chunks = self._chunk_text(text, chunk_size=300, overlap=50)

        for chunk_idx, chunk in enumerate(chunks):
            passages.append(
                LegalPassage(
                    passage_id=f"contract_{doc_idx}_{chunk_idx}", 
                    text = chunk,
                    source="ContractNLI", 
                    case_namef"Contract Dcoument {doc_idx}",
                )
            )

            if len(passages) >= self.max_passages:
                break
        
    if not passages:
        raise ValueError("No passages were created from ContractNLI.")
    
    return passages

@staticmethod
def _extract_documents(corpus):
    if isinstance(corpus, list):
        return corpus
    
    if isinstance(corpus, dict):
        if "documents" in corpus:
            return corpus["documents"]
        if "data" in corpus:
            return corpus["data"]
        if "premises" in corpus:
            return corpus["premises"]
    
    return []

@staticmethod
def _tract_text(doc):
    if isinstance(doc, str):
        return doc
    
    if not isinstance(doc, dict):
        return ""
    
    possible_keys = [
        "text",
        "contract_text", 
        "premise", 
        "document", 
        "content",
    ]

    for key in possible_keys:
        if key in doc and isinstance(doc[key], str):
            return doc[key]

    return ""

@staticmethod
def _chunk_text(text:str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        
        if end == len(words):
            break

        start += chunk_size - overlap

    return chunks

class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = sentenceTransfomer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
    
    def encode(self, text: List[str], batch_size: int = 32) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size, 
            show_progress_bar=False, 
            convert_to_numpy=True,
        )

        return embeddings.astype(np.float32)




