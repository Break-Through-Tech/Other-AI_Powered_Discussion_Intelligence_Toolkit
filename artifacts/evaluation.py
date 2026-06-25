"""
Evaluation harness — metric functions, model protocols, evaluate() helpers.

The harness is intentionally minimal:

  * Protocols for the model interfaces students must satisfy
  * Pure metric functions (sklearn / scipy wrappers, no surprises)
  * Pydantic result types so outputs are validated and serializable
  * evaluate_* helpers that pair a model with a labeled split and return results

The harness does NOT:

  * Provide training loops (student work, October milestone)
  * Provide model implementations (these are the empty slots students fill)
  * Track experiments across runs (use a notebook or wandb if needed)

Design notes:

  * Models are `Protocol` types, not ABCs. Students can satisfy them with any
    object that has a `.predict()` method — sklearn pipelines, HF model wrappers,
    LLM-backed callables. No inheritance required.

  * Metrics are pure functions over parallel sequences. They don't know about
    models, datasets, or splits. evaluate_* helpers wire models to metrics.

  * Results are Pydantic models so they round-trip cleanly through JSON.
"""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np
from pydantic import BaseModel, Field
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    silhouette_score,
)


# ---------------------------------------------------------------------------
# Model protocols  (empty slots — students implement)
# ---------------------------------------------------------------------------


class TextClassifier(Protocol):
    """Anything that maps a list of texts to a list of class labels.

    Satisfied by sklearn pipelines, Hugging Face transformers wrappers,
    LLM-backed callables, dummy baselines, anything with a `predict` method.
    """

    def predict(self, texts: Sequence[str]) -> list[str]: ...


class TextRegressor(Protocol):
    """Anything that maps a list of texts to a list of scalar scores."""

    def predict(self, texts: Sequence[str]) -> list[float]: ...


class TextEmbedder(Protocol):
    """Anything that maps a list of texts to a (N, D) array of embeddings."""

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class ClassificationResult(BaseModel):
    """Metrics from evaluating a TextClassifier against gold labels."""

    n_examples: int
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class_f1: dict[str, float]
    confusion: list[list[int]] = Field(description="Confusion matrix, rows indexed by `labels`")
    labels: list[str] = Field(description="Sorted label inventory used for the confusion matrix")


class RegressionResult(BaseModel):
    """Metrics from evaluating a TextRegressor against gold scores."""

    n_examples: int
    mae: float
    spearman_r: float
    spearman_p: float


class ClusteringResult(BaseModel):
    """Metrics from evaluating an embedding + clustering against itself."""

    n_examples: int
    n_clusters: int
    silhouette: float = Field(description="Mean silhouette coefficient; NaN if not computable")


# ---------------------------------------------------------------------------
# Pure metric functions
# ---------------------------------------------------------------------------


def classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
) -> ClassificationResult:
    """Compute accuracy, macro/weighted F1, per-class F1, and confusion matrix."""
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: {len(y_true)} gold vs {len(y_pred)} predictions")
    labels = sorted(set(y_true) | set(y_pred))
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return ClassificationResult(
        n_examples=len(y_true),
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        per_class_f1={lbl: float(score) for lbl, score in zip(labels, per_class)},
        confusion=cm.tolist(),
        labels=labels,
    )


def regression_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> RegressionResult:
    """Compute MAE and Spearman rank correlation."""
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: {len(y_true)} gold vs {len(y_pred)} predictions")
    rho, p = spearmanr(y_true, y_pred)
    return RegressionResult(
        n_examples=len(y_true),
        mae=float(mean_absolute_error(y_true, y_pred)),
        spearman_r=float(rho) if not np.isnan(rho) else 0.0,
        spearman_p=float(p) if not np.isnan(p) else 1.0,
    )


def clustering_metrics(
    embeddings: np.ndarray,
    cluster_labels: Sequence[int],
) -> ClusteringResult:
    """Compute silhouette score; NaN if not computable (fewer than 2 clusters or singleton clusters)."""
    n_clusters = len(set(cluster_labels))
    can_compute = 2 <= n_clusters < len(embeddings)
    sil = float(silhouette_score(embeddings, cluster_labels)) if can_compute else float("nan")
    return ClusteringResult(
        n_examples=len(embeddings),
        n_clusters=n_clusters,
        silhouette=sil,
    )


def retrieval_recall_at_k(
    relevant_ids: Sequence[set[str]],
    retrieved_ids: Sequence[list[str]],
    k: int,
) -> float:
    """Mean recall@k across multiple queries.

    Args:
        relevant_ids: For each query, the set of relevant doc IDs (ground truth).
        retrieved_ids: For each query, the ranked list of retrieved doc IDs.
        k: Cut-off depth for retrieval.

    Returns:
        Mean recall@k. Queries with empty relevant sets are skipped.
    """
    if len(relevant_ids) != len(retrieved_ids):
        raise ValueError("relevant_ids and retrieved_ids must be parallel")
    recalls: list[float] = []
    for relevant, retrieved in zip(relevant_ids, retrieved_ids):
        if not relevant:
            continue
        top_k = set(retrieved[:k])
        recalls.append(len(top_k & relevant) / len(relevant))
    return float(np.mean(recalls)) if recalls else 0.0


# ---------------------------------------------------------------------------
# evaluate_* helpers — pair a model with parallel data, return result
# ---------------------------------------------------------------------------


def evaluate_classifier(
    model: TextClassifier,
    texts: Sequence[str],
    labels: Sequence[str],
) -> ClassificationResult:
    """Run a classifier over texts and compute metrics against gold labels."""
    predictions = model.predict(texts)
    if len(predictions) != len(labels):
        raise ValueError(
            f"Model returned {len(predictions)} predictions for {len(labels)} examples"
        )
    return classification_metrics(labels, predictions)


def evaluate_regressor(
    model: TextRegressor,
    texts: Sequence[str],
    scores: Sequence[float],
) -> RegressionResult:
    """Run a regressor over texts and compute metrics against gold scores."""
    predictions = model.predict(texts)
    if len(predictions) != len(scores):
        raise ValueError(
            f"Model returned {len(predictions)} predictions for {len(scores)} examples"
        )
    return regression_metrics(scores, predictions)
