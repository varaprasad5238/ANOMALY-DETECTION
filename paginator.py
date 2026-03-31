"""
paginator.py
============
Pagination for a dynamically ranked dataset aggregated from multiple external APIs.

Problem
-------
Applying offset/limit pagination is straightforward when data comes from a single
source with a stable order.  It becomes hard when:

* Records are fetched from **multiple independent external APIs** whose internals
  are unknown (different counts, orderings, pagination schemes).
* Final ranking is determined **after** aggregation, by an ML model loaded from a
  pickle file.

Applying offset/limit per-API before sorting would skip or duplicate records
because the ML-based ranking can completely re-order the merged dataset.

Solution
--------
Treat every API's response as a shard of one virtual dataset:

1. Fetch **all** records from every API.
2. Merge into a single list.
3. Apply global ML-model scoring / custom sort key.
4. Slice with ``offset`` and ``limit``.

This guarantees that records ranked highly by the ML model are never accidentally
excluded from early pages.

Usage
-----
::

    from paginator import AggregatedPaginator

    def fetch_industry_a():
        # call external API A and return list of dicts
        return [{"id": 1, "value": 3.2}, ...]

    def fetch_industry_b():
        # call external API B and return list of dicts
        return [{"id": 2, "value": 7.1}, ...]

    paginator = AggregatedPaginator(
        api_fetchers=[fetch_industry_a, fetch_industry_b],
        model_path="ranking_model.pkl",   # optional
    )

    paginator.load_and_sort(feature_keys=["value"])

    page = paginator.paginate(offset=0, limit=10)
    # {"total": <int>, "offset": 0, "limit": 10, "data": [...]}
"""

import pickle
from typing import Any, Callable, Dict, List, Optional

import numpy as np


class AggregatedPaginator:
    """Aggregate records from multiple external APIs, rank them, and paginate.

    Parameters
    ----------
    api_fetchers:
        Iterable of zero-argument callables.  Each callable represents one
        external API and must return a list of record dicts.
    model_path:
        Optional path to a pickle file that contains a trained scikit-learn
        (or compatible) model used to score records for ranking.  The model
        must expose either ``predict_proba`` (classification) or ``predict``
        (regression / scoring) methods.
    score_field:
        Name of the key added to each record dict to store the computed score.
        Defaults to ``"score"``.
    """

    def __init__(
        self,
        api_fetchers: List[Callable[[], List[Dict]]],
        model_path: Optional[str] = None,
        score_field: str = "score",
    ) -> None:
        self.api_fetchers = api_fetchers
        self.model_path = model_path
        self.score_field = score_field

        self._model = None
        self._sorted_data: Optional[List[Dict]] = None

        if model_path:
            self._load_model(model_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self, model_path: str) -> None:
        """Deserialise the ML model from *model_path*."""
        with open(model_path, "rb") as fh:
            self._model = pickle.load(fh)

    def _fetch_all(self) -> List[Dict]:
        """Call every API fetcher and merge results into one list."""
        merged: List[Dict] = []
        for fetcher in self.api_fetchers:
            records = fetcher()
            merged.extend(records)
        return merged

    def _compute_scores(
        self, records: List[Dict], feature_keys: List[str]
    ) -> List[Dict]:
        """Add a score to each record using the loaded ML model.

        Parameters
        ----------
        records:
            Merged list of record dicts.
        feature_keys:
            Keys whose values are used as the feature vector fed to the model.

        Returns
        -------
        The same list of records, each enriched with ``self.score_field``.
        """
        if not records:
            return records

        feature_matrix = np.array(
            [[record[k] for k in feature_keys] for record in records],
            dtype=float,
        )

        if hasattr(self._model, "predict_proba"):
            scores = self._model.predict_proba(feature_matrix)[:, 1]
        elif hasattr(self._model, "predict"):
            scores = self._model.predict(feature_matrix)
        else:
            raise ValueError(
                "The loaded model must implement predict_proba() or predict()."
            )

        for record, score in zip(records, scores):
            record[self.score_field] = float(score)

        return records

    def _sort_by_score(
        self, records: List[Dict], ascending: bool
    ) -> List[Dict]:
        """Sort records by ``self.score_field``."""
        return sorted(
            records,
            key=lambda r: r.get(self.score_field, 0.0),
            reverse=not ascending,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_and_sort(
        self,
        feature_keys: Optional[List[str]] = None,
        sort_key: Optional[Callable[[Dict], Any]] = None,
        ascending: bool = False,
    ) -> "AggregatedPaginator":
        """Fetch all data from every API, score it, and sort globally.

        Call this method **once** before calling :meth:`paginate`.  If the
        underlying data changes you can call it again to refresh.

        Parameters
        ----------
        feature_keys:
            Column names used to build the feature matrix for the ML model.
            Required when *model_path* was supplied at construction time.
        sort_key:
            Optional Python callable used as the ``key`` argument of
            :py:func:`sorted`.  When provided it takes precedence over the ML
            model, which lets you supply a simple lambda or a custom ranking
            function without a pickle model.
        ascending:
            ``True`` → lowest score first.  ``False`` (default) → highest
            score first (top-ranked records appear on page 1).

        Returns
        -------
        *self* — so calls can be chained::

            page = paginator.load_and_sort(feature_keys=["v"]).paginate(0, 10)
        """
        records = self._fetch_all()

        if sort_key is not None:
            self._sorted_data = sorted(
                records, key=sort_key, reverse=not ascending
            )
        elif self._model is not None and feature_keys:
            records = self._compute_scores(records, feature_keys)
            self._sorted_data = self._sort_by_score(records, ascending)
        else:
            # No model and no sort_key: preserve merged order.
            self._sorted_data = records

        return self

    def paginate(self, offset: int = 0, limit: int = 10) -> Dict:
        """Return one page from the globally sorted dataset.

        Parameters
        ----------
        offset:
            Number of records to skip from the start of the sorted list.
            Must be >= 0.
        limit:
            Maximum number of records to include in the page.  Must be > 0.

        Returns
        -------
        A dict with the following keys:

        ``total``
            Total number of records across all APIs (after de-duplication, if
            any, happens in the fetchers).
        ``offset``
            The requested offset (echoed back for client convenience).
        ``limit``
            The requested limit (echoed back).
        ``data``
            List of record dicts for this page.  May be shorter than *limit*
            when the offset is near the end of the dataset.

        Raises
        ------
        RuntimeError
            If :meth:`load_and_sort` has not been called yet.
        ValueError
            If *offset* or *limit* are out of range.
        """
        if self._sorted_data is None:
            raise RuntimeError(
                "Call load_and_sort() before paginate()."
            )
        if offset < 0:
            raise ValueError(f"offset must be >= 0, got {offset}.")
        if limit <= 0:
            raise ValueError(f"limit must be > 0, got {limit}.")

        total = len(self._sorted_data)
        page_data = self._sorted_data[offset: offset + limit]

        return {
            "total": total,
            "offset": offset,
            "limit": limit,
            "data": page_data,
        }
