"""Exact multiset Weighted Jaccard search using sparse token-count postings."""
from __future__ import annotations

from array import array
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix

from .metrics import tokenize


class WeightedNeighborIndex:
    """Accumulate integer count vectors, then search every indexed rollout."""

    def __init__(self):
        self.vocabulary = {}
        self.indices, self.data, self.indptr = array("i"), array("i"), array("q", [0])
        self.lengths = []
        self.postings = None

    def add(self, counts: Counter) -> int:
        if self.postings is not None:
            raise RuntimeError("Cannot add documents after finalizing index")
        length = sum(counts.values())
        for token, count in counts.items():
            index = self.vocabulary.setdefault(token, len(self.vocabulary))
            self.indices.append(index)
            self.data.append(count)
        self.indptr.append(len(self.indices))
        self.lengths.append(length)
        return length

    def finalize(self):
        matrix = csr_matrix((np.frombuffer(self.data, dtype=np.int32),
                             np.frombuffer(self.indices, dtype=np.int32),
                             np.frombuffer(self.indptr, dtype=np.int64)),
                            shape=(len(self.lengths), len(self.vocabulary)))
        self.postings = matrix.tocsc()
        self.lengths = np.asarray(self.lengths, dtype=np.int64)
        self.indices = self.data = self.indptr = None

    def query(self, counts: Counter, k=5):
        if self.postings is None:
            raise RuntimeError("Finalize index before querying")
        if not 1 <= k <= len(self.lengths):
            raise ValueError("Not enough training trajectories for requested K")
        intersection = np.zeros(len(self.lengths), dtype=np.int64)
        for token, count in counts.items():
            column = self.vocabulary.get(token)
            if column is None:
                continue
            start, end = self.postings.indptr[column:column+2]
            documents = self.postings.indices[start:end]
            intersection[documents] += np.minimum(count, self.postings.data[start:end])
        union = sum(counts.values()) + self.lengths - intersection
        similarities = np.divide(intersection, union, out=np.ones(len(union), dtype=float), where=union != 0)
        order = np.lexsort((np.arange(len(similarities)), -similarities))[:k+1]
        return {"neighbors": [{"index": int(i), "similarity": float(similarities[i])} for i in order[:k]],
                "boundary_tie": bool(len(order) > k and similarities[order[k-1]] == similarities[order[k]])}


def token_counts(text):
    return Counter(tokenize(text))
