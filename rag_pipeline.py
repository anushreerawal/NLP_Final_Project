
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


