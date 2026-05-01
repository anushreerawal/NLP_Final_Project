
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
    source: str
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
    latency_ms: float = 0.0

# Dataset loader for ContractNLI

class ContractNLILoader:
    URL = "https://stanfordnlp.github.io/contract-nli/resources/contract-nli.zip"

    def __init__(self, max_passages: int = 500):
        self.max_passages = max_passages
    

    def load_corpus(self) -> List[LegalPassage]:
        import os
        cache_path = "contract_nli_cache.zip"

        if os.path.exists(cache_path):
            print("[INFO] Loading ContractNLI from local cache...")
            with open(cache_path, "rb") as f:
                data = f.read()
        else:
            print("[INFO] Downloading ContractNLI dataset (one-time, ~30MB)...")
            with urllib.request.urlopen(self.URL, timeout=120) as response:
                data = response.read()
            with open(cache_path, "wb") as f:
                f.write(data)
            print(f"[INFO] Saved to cache ({len(data)/1024/1024:.1f}MB)")

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
            print(f"[INFO] Using: {train_file}")

            with zf.open(train_file) as f:
                corpus = json.load(f)

        docs = self._extract_documents(corpus)

        if not docs:
            raise ValueError("Could not find documents inside ContractNLI data.")

        print(f"[INFO] Found {len(docs)} documents, chunking...")

        for doc_idx, doc in enumerate(docs):
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
                        text=chunk,
                        source="ContractNLI",
                        case_name=f"Contract Document {doc_idx}",
                    )
                )

                if len(passages) >= self.max_passages:
                    break

        if not passages:
            raise ValueError("No passages were created from ContractNLI.")

        print(f"[INFO] Loaded {len(passages)} passages from ContractNLI.")
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
    def _extract_text(doc):
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
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_embedding_dimension()
    
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size, 
            show_progress_bar=False, 
            convert_to_numpy=True,
        )

        return embeddings.astype(np.float32)
    
class VectorIndex:

    def __init__(self, dim:int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.passages: List[LegalPassage] = []
    
    def build(self, passages: List[LegalPassage], embeddings:np.ndarray):
        if len(passages) != embeddings.shape[0]:
            raise ValueError("Number of passages does not match number of embeddings.")
        
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
        normalized_embeddings = embeddings / norms

        self.passages = passages
        self.index.add(normalized_embeddings)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[LegalPassage]:
        query = query_embedding.reshape(1, -1).astype(np.float32)
        query = query / (np.linalg.norm(query) + 1e-9)

        scores, indicies = self.index.search(query, top_k)

        results = []

        for score, idx in zip(scores[0], indicies[0]):
            if idx == -1:
                continue

            passage = self.passages[idx]
            passage.score = float(score)
            results.append(passage)
        
        return results
    
class AnswerGenerator:
    def generate(self, question: str, passages: List[LegalPassage]) -> str:
        if not passages:
            return "No relevant legal passage was found."
        
        top_passage = passages[0]

        return (
            "Based on the retrieved passage from ContractNLI, the relevant text says:\n\n"
            f"{top_passage.text[:700]}"
        )

class LegalRAGPipeline:
    def __init__(
        self,
        sbert_model: str = "all-MiniLM-L6-v2", 
        top_k: int = 5, 
        max_passages: int = 500, 
    ):
        self.top_k = top_k
        self.loader = ContractNLILoader(max_passages=max_passages)
        self.embedder = EmbeddingModel(sbert_model)
        self.index = VectorIndex(self.embedder.dim)
        self.generator = AnswerGenerator()
        self.built = False
    
    def build_index(self):
        corpus = self.loader.load_corpus()
        texts = [passage.text for passage in corpus]
        embeddings = self.embedder.encode(texts)

        self.index.build(corpus, embeddings)
        self.built = True

    def answer(self, question: str) -> RAGResult:
        if not self.built:
            self.build_index()
        
        start_time = time.time()

        question_embedding = self.embedder.encode([question])[0]
        retrieved_passages = self.index.search(question_embedding, self.top_k)
        generated_answer = self.generator.generate(question, retrieved_passages)

        latency_ms = (time.time() - start_time) * 1000

        return RAGResult(
            question=question, 
            retrieved_passages=retrieved_passages, 
            generated_answer=generated_answer,
            retrieval_score=0,
            answer_score=0,
            latency_ms=latency_ms,
        )
    
if __name__ == "__main__":
    pipeline = LegalRAGPipeline(
        sbert_model="all-MiniLM-L6-v2",
        top_k=3,
        max_passages=500,
    )

    pipeline.build_index()

    question = "Can the receiving party disclose confidential information to third parties?"
    result = pipeline.answer(question)

    print(f"\nQuestion: {result.question}")
    print(f"\nAnswer:\n{result.generated_answer}")
    print(f"\nLatency: {result.latency_ms:.0f}ms")
    print(f"\nTop retrieved passage (score={result.retrieved_passages[0].score:.3f}):")
    print(result.retrieved_passages[0].text[:300])
        


