"""Similarity metrics in [0, 1] between evaluation and training trajectories, over lowercased word/code tokens.

Dense metrics return the full (n_eval, n_train) matrix; rerank metrics are only scored on the top cand_k candidates.
"""

from __future__ import annotations

import re
from typing import Callable

import numpy as np


_TOK_RE = re.compile(r"[A-Za-z_]\w+|\d+|[^\sA-Za-z0-9_]")


def tokenize(text: str, lower: bool = True) -> list[str]:
    if lower:
        text = text.lower()
    return _TOK_RE.findall(text)


def word_ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if len(tokens) < n:
        return [tuple(tokens)] if tokens else []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


class Corpus:
    """Trajectory texts with cached token representations."""

    def __init__(self, texts: list[str]):
        self.texts = texts
        self._tokens: list[list[str]] | None = None
        self._token_sets: list[set] | None = None
        self._word_sets: dict[int, list[set]] = {}

    def __len__(self) -> int:
        return len(self.texts)

    def tokens(self) -> list[list[str]]:
        if self._tokens is None:
            self._tokens = [tokenize(t) for t in self.texts]
        return self._tokens

    def token_sets(self) -> list[set]:
        """Unique-token sets."""
        if self._token_sets is None:
            self._token_sets = [set(t) for t in self.tokens()]
        return self._token_sets

    def word_ngram_sets(self, n: int) -> list[set]:
        if n not in self._word_sets:
            self._word_sets[n] = [set(word_ngrams(toks, n)) for toks in self.tokens()]
        return self._word_sets[n]

DenseFn = Callable[[Corpus, Corpus], np.ndarray]
RerankFn = Callable[[list, list], float]

DENSE_METRICS: dict[str, DenseFn] = {}
RERANK_METRICS: dict[str, RerankFn] = {}


def dense_metric(name: str):
    def deco(fn: DenseFn) -> DenseFn:
        DENSE_METRICS[name] = fn
        return fn

    return deco


def rerank_metric(name: str):
    def deco(fn: RerankFn) -> RerankFn:
        RERANK_METRICS[name] = fn
        return fn

    return deco


def all_metric_names() -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for n in list(DENSE_METRICS) + list(RERANK_METRICS):
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _all_pairs_set_sim(eval_sets: list[set], train_sets: list[set], kind: str) -> np.ndarray:
    """(n_eval, n_train) set similarity via a sparse matmul."""
    from scipy.sparse import csr_matrix

    vocab: dict = {}
    for s in eval_sets:
        for x in s:
            vocab.setdefault(x, len(vocab))
    for s in train_sets:
        for x in s:
            vocab.setdefault(x, len(vocab))
    V = len(vocab)

    def incidence(sets: list[set]) -> csr_matrix:
        rows, cols = [], []
        for i, s in enumerate(sets):
            for x in s:
                rows.append(i)
                cols.append(vocab[x])
        data = np.ones(len(rows), dtype=np.float32)
        return csr_matrix((data, (rows, cols)), shape=(len(sets), max(V, 1)))

    E = incidence(eval_sets)
    T = incidence(train_sets)
    inter = (E @ T.T).toarray()
    e_sz = np.array([len(s) for s in eval_sets], dtype=np.float32)[:, None]
    t_sz = np.array([len(s) for s in train_sets], dtype=np.float32)[None, :]

    if kind == "jaccard":
        union = e_sz + t_sz - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            sim = np.where(union > 0, inter / union, 0.0)
    elif kind == "containment":
        with np.errstate(divide="ignore", invalid="ignore"):
            sim = np.where(e_sz > 0, inter / e_sz, 0.0)
    else:
        raise ValueError(kind)
    return sim.astype(np.float32)


@dense_metric("token_jaccard")
def _token_jaccard(ec: Corpus, tc: Corpus) -> np.ndarray:
    """Set Jaccard over unique tokens: |A ∩ B| / |A ∪ B|."""
    return _all_pairs_set_sim(ec.token_sets(), tc.token_sets(), "jaccard")


@dense_metric("token_containment")
def _token_containment(ec: Corpus, tc: Corpus) -> np.ndarray:
    """Fraction of eval's unique tokens that appear in train: |A ∩ B| / |A|."""
    return _all_pairs_set_sim(ec.token_sets(), tc.token_sets(), "containment")


@dense_metric("word3_jaccard")
def _word3_jaccard(ec: Corpus, tc: Corpus) -> np.ndarray:
    return _all_pairs_set_sim(ec.word_ngram_sets(3), tc.word_ngram_sets(3), "jaccard")


@dense_metric("word3_containment")
def _word3_containment(ec: Corpus, tc: Corpus) -> np.ndarray:
    return _all_pairs_set_sim(ec.word_ngram_sets(3), tc.word_ngram_sets(3), "containment")


# not plotted; used to pick rerank candidates
def _tfidf_cosine(eval_texts, train_texts, analyzer, ngram_range, min_df=2) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vec = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        min_df=min_df,
        lowercase=True,
        token_pattern=r"(?u)[A-Za-z_]\w+|\d+|[^\sA-Za-z0-9_]" if analyzer == "word" else None,
    )
    T = vec.fit_transform(train_texts)
    E = vec.transform(eval_texts)
    return cosine_similarity(E, T).astype(np.float32)


@dense_metric("tfidf_word12")
def _tfidf_word12(ec: Corpus, tc: Corpus) -> np.ndarray:
    return _tfidf_cosine(ec.texts, tc.texts, "word", (1, 2))


@rerank_metric("token_levenshtein")
def _token_levenshtein(a: list[str], b: list[str]) -> float:
    from rapidfuzz.distance import Levenshtein

    if not a and not b:
        return 1.0
    return float(Levenshtein.normalized_similarity(a, b))


def weighted_jaccard_pair(a: list[str], b: list[str]) -> float:
    """Multiset (count-weighted) token Jaccard: Σ_x min(cA,cB) / Σ_x max(cA,cB)."""
    from collections import Counter

    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    ca, cb = Counter(a), Counter(b)
    inter = sum((ca & cb).values())
    union = sum((ca | cb).values())
    return inter / union if union else 0.0


# pairwise form, for the shortlist
rerank_metric("weighted_jaccard")(weighted_jaccard_pair)


@dense_metric("weighted_jaccard")
def _weighted_jaccard_dense(ec: Corpus, tc: Corpus) -> np.ndarray:
    """Exact all-pairs Ruzicka (weighted Jaccard), so the nearest neighbour is exact."""
    from collections import Counter

    e_toks, t_toks = ec.tokens(), tc.tokens()
    t_counts = [Counter(t) for t in t_toks]
    t_len = np.array([len(t) for t in t_toks], dtype=np.int64)
    out = np.zeros((len(e_toks), len(t_toks)), dtype=np.float32)
    for i, et in enumerate(e_toks):
        ca = Counter(et)
        e_len = len(et)
        if e_len == 0:
            out[i] = np.where(t_len == 0, 1.0, 0.0)
            continue
        ca_items = ca.items()
        for j, cb in enumerate(t_counts):
            if t_len[j] == 0:
                continue
            inter = 0
            for tok, cv in ca_items:
                tv = cb.get(tok)
                if tv:
                    inter += cv if cv < tv else tv
            union = e_len + t_len[j] - inter
            if union:
                out[i, j] = inter / union
    return out


@rerank_metric("len_ratio")
def _len_ratio(a: list[str], b: list[str]) -> float:
    """Length ratio min(len) / max(len), ignoring content."""
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    return min(la, lb) / max(la, lb) if max(la, lb) else 0.0


CANDIDATE_SOURCES = ("tfidf_word12", "word3_jaccard")
TOKEN_CANDIDATE_SOURCES = ("token_jaccard", "token_containment")


def candidate_sources_for(metric_names: list[str]) -> tuple[str, ...]:
    """Use token-overlap candidates when every requested metric is token-wise."""
    tokenish = {
        "token_jaccard", "token_containment", "token_levenshtein",
        "token_lcs", "weighted_jaccard", "len_ratio",
    }
    if metric_names and all(m in tokenish for m in metric_names):
        return TOKEN_CANDIDATE_SOURCES
    return CANDIDATE_SOURCES
