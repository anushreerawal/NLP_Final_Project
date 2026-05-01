import math
import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

import numpy as np


STOP_WORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "he", "him", "his", "she", "her", "hers", "it", "its", "they",
    "them", "their", "what", "which", "who", "whom", "this", "that", "these",
    "those", "am", "is", "are", "was", "were", "be", "been", "being", "have",
    "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "must", "shall", "can", "a", "an", "the", "and", "but",
    "or", "nor", "for", "so", "yet", "at", "by", "in", "of", "on", "to",
    "up", "as", "into", "with", "about", "between", "through",
}

def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-z]+", text.lower())
    return [t for t in tokens if t not in STOP_WORDS]


#baseline 1 - Majority Class
class MajorityClassBaseline:
    
    def __init__(self):
        self._majority_answer = "unknown"

    def fit(self, examples: List[QAExample]):
        all_tokens = []
        for ex in examples:
            all_tokens.extend(tokenize(ex.answer))
        if all_tokens:
            self._majority_answer = Counter(all_tokens).most_common(1)[0][0]

    def predict(self, question: str) -> str:
        return self._majority_answer 
    
    def evaluate(self, examples: List[QAExample], corpus: List[LegalPassage]) -> List[RAGResult]:
        results = []
        for ex in examples:
            answer = self.predict(ex.question)
            results.append(RAGResult(
                question=ex.question,
                retrieved_passages=[],  
                generated_answer=answer,
                retrieval_score=0,
                answer_score=self._score(answer, ex.answer),
            ))
        return results
    
    @staticmethod 
    def _score(generated: str, reference: str) -> int:
        gen_toks = set(tokenize(generated))
        ref_toks = set(tokenize(reference))
        if not ref_toks:
            return 0
        overlap = len(gen_toks & ref_toks) / len(ref_toks)
        return 2 if overlap >= 0.5 else (1 if overlap >= 0.25 else 0)
    
    #baseline 2 - BM25 Keyword Retrieval

    class BM25Retriever:
   

        def __init__(self, k1: float = 1.5, b: float = 0.75):
            self.k1 = k1
            self.b = b
            self._passages: List[LegalPassage] = []
            self._tf: List[Dict[str, int]] = []
            self._df: Dict[str, int] = defaultdict(int)
            self._avg_dl: float = 0.0
            self._N: int = 0

        def fit(self, passages: List[LegalPassage]):
            self._passages = passages
            self._N = len(passages)
            doc_lengths = []
            self._tf = []

            for p in passages:
                tokens = tokenize(p.text)
                doc_lengths.append(len(tokens))
                tf = Counter(tokens)
                self._tf.append(tf)
                for term in tf:
                    self._df[term] += 1

        self._avg_dl = sum(doc_lengths) / max(self._N, 1)


        def search(self, query: str, top_k: int = 5) -> List[LegalPassage]:
        
            q_tokens = tokenize(query)
            scores = np.zeros(self._N)

            for term in q_tokens:
                df = self._df.get(term, 0)
                if df == 0:
                    continue
                idf = math.log((self._N - df + 0.5) / (df + 0.5) + 1)
                for i, tf in enumerate(self._tf):
                    freq = tf.get(term, 0)
                    dl = sum(tf.values())
                    numerator = freq * (self.k1 + 1)
                    denominator = freq + self.k1 * (
                        1 - self.b + self.b * dl / self._avg_dl
                    )
                    scores[i] += idf * (numerator / denominator)

            top_idx = np.argsort(scores)[::-1][:top_k]
            results = []
            for idx in top_idx:
                p = self._passages[idx]
                p.score = float(scores[idx])
                results.append(p)
            return results
        
        def evaluate(self, examples: List[QAExample], top_k: int = 5) -> List[RAGResult]:
            results = []
            for ex in examples:
                passages = self.search(ex.question, top_k)
                answer = passages[0].text[:300] if passages else "No passage found."
                if ex.source_passage_ids:
                    retrieved_ids = {p.passage_id for p in passages}
                    retrieval_score = int(bool(retrieved_ids & set(ex.source_passage_ids)))
                else:
                    import re as _re
                    stops = {"the","a","an","is","of","and","or","in","to",
                            "be","that","this","it","for","on","with","as",
                            "are","by","no","yes","not","only","any","all"}
                    def _kw(t): return {w for w in t.lower().split() if w.isalpha() and w not in stops}
                    ret_kw = _kw(" ".join(p.text for p in passages))
                    ref_kw = _kw(ex.answer)
                    retrieval_score = int(bool(ref_kw) and len(ref_kw & ret_kw)/len(ref_kw) >= 0.30)
                results.append(RAGResult(
                    question=ex.question,
                    retrieved_passages=passages,
                    generated_answer=answer,
                    retrieval_score=retrieval_score,
                    answer_score=self._score_answer(answer, ex.answer),
                ))
            return results

        @staticmethod
        def _score_answer(generated: str, reference: str) -> int:
            gen_toks = set(tokenize(generated))
            ref_toks = set(tokenize(reference))
            if not ref_toks:
                return 0
            overlap = len(gen_toks & ref_toks) / len(ref_toks)
            return 2 if overlap >= 0.5 else (1 if overlap >= 0.25 else 0)
        














