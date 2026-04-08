from __future__ import annotations

import structlog
import numpy as np
import re
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from app.agents.state import AnalysisState
from app.models.schemas import Cluster, LogEvent, LogLevel

logger = structlog.get_logger()

# Only cluster events at WARNING and above
_CLUSTER_LEVELS = {LogLevel.WARNING, LogLevel.ERROR, LogLevel.CRITICAL}

# DBSCAN defaults — tunable via caller if needed
_EPS = 0.35
_MIN_SAMPLES = 2


def normalize_message(message: str) -> str:
    # Remplace tous les nombres par un token générique
    return re.sub(r'\d+', 'NUM', message)


def cluster_events(events: list[LogEvent], eps: float = _EPS, min_samples: int = _MIN_SAMPLES) -> list[Cluster]:
    """
    Vectorise event messages with TF-IDF, run DBSCAN, return Cluster list.
    Noise points (label -1) are each returned as their own singleton cluster.
    """
    candidates = [e for e in events if e.level in _CLUSTER_LEVELS]

    if not candidates:
        return []

    messages = [normalize_message(e.message) for e in candidates]

    # TF-IDF on character n-grams captures token sub-structure well for log messages
    vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1)
    X = vectorizer.fit_transform(messages)
    X_norm = normalize(X, norm="l2")

    labels: np.ndarray = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit_predict(X_norm)

    # Group indices by label
    clusters: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(idx)

    result: list[Cluster] = []
    cluster_id = 0

    for label, indices in sorted(clusters.items()):
        if label == -1:
            # Noise: one singleton cluster per point
            for i in indices:
                result.append(Cluster(
                    cluster_id=cluster_id,
                    size=1,
                    representative=candidates[i].message,
                ))
                cluster_id += 1
        else:
            # Pick the message closest to the centroid as representative
            sub = X_norm[indices]
            centroid = np.asarray(sub.mean(axis=0))
            sims = sub.dot(centroid.T).A1 if hasattr(sub, "A1") else sub.dot(centroid.T).flatten()
            rep_idx = indices[int(np.argmax(sims))]
            result.append(Cluster(
                cluster_id=cluster_id,
                size=len(indices),
                representative=candidates[rep_idx].message,
            ))
            cluster_id += 1

    return result


def pattern_agent(state: AnalysisState) -> AnalysisState:
    """Cluster error/warning events with DBSCAN and populate state.clusters."""
    clusters = cluster_events(state["events"])
    logger.info(
        "pattern.done",
        n_events=len(state["events"]),
        n_clusters=len(clusters),
    )
    return {**state, "clusters": clusters}
