"""
test_paginator.py
=================
Unit tests for AggregatedPaginator.

Run with:
    python -m pytest test_paginator.py -v
or simply:
    python test_paginator.py
"""

import pickle
import tempfile
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from paginator import AggregatedPaginator


# ---------------------------------------------------------------------------
# Helpers / stub "external APIs"
# ---------------------------------------------------------------------------

def make_api(records):
    """Return a zero-argument callable that yields *records*."""
    def _fetcher():
        return list(records)
    return _fetcher


# Module-level fake model classes so they can be serialised with pickle.

class FakeModel:
    """Minimal model with predict(); returns pre-set scores in order."""

    def __init__(self, scores):
        self._scores = list(scores)

    def predict(self, X):
        return np.array(self._scores[: len(X)])


class FakeProbaModel:
    """Minimal model with predict_proba(); returns [[1-p, p], …]."""

    def __init__(self, scores):
        self._scores = list(scores)

    def predict_proba(self, X):
        col1 = np.array(self._scores[: len(X)])
        col0 = 1.0 - col1
        return np.column_stack([col0, col1])


def make_model(scores):
    """Return a FakeModel instance."""
    return FakeModel(scores)


def make_proba_model(scores):
    """Return a FakeProbaModel instance."""
    return FakeProbaModel(scores)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFetchAndMerge(unittest.TestCase):
    """load_and_sort fetches from all APIs and merges results."""

    def test_single_api(self):
        records = [{"id": i} for i in range(5)]
        pag = AggregatedPaginator(api_fetchers=[make_api(records)])
        pag.load_and_sort()
        result = pag.paginate(offset=0, limit=10)
        self.assertEqual(result["total"], 5)
        self.assertEqual(len(result["data"]), 5)

    def test_multiple_apis_merged(self):
        api_a = [{"id": 1, "src": "A"}, {"id": 2, "src": "A"}]
        api_b = [{"id": 3, "src": "B"}]
        api_c = [{"id": 4, "src": "C"}, {"id": 5, "src": "C"}, {"id": 6, "src": "C"}]
        pag = AggregatedPaginator(
            api_fetchers=[make_api(api_a), make_api(api_b), make_api(api_c)]
        )
        pag.load_and_sort()
        result = pag.paginate(offset=0, limit=20)
        self.assertEqual(result["total"], 6)
        ids = {r["id"] for r in result["data"]}
        self.assertEqual(ids, {1, 2, 3, 4, 5, 6})

    def test_empty_apis(self):
        pag = AggregatedPaginator(api_fetchers=[make_api([]), make_api([])])
        pag.load_and_sort()
        result = pag.paginate(offset=0, limit=10)
        self.assertEqual(result["total"], 0)
        self.assertEqual(result["data"], [])

    def test_mixed_empty_and_nonempty(self):
        pag = AggregatedPaginator(
            api_fetchers=[make_api([]), make_api([{"id": 1}])]
        )
        pag.load_and_sort()
        result = pag.paginate(offset=0, limit=5)
        self.assertEqual(result["total"], 1)


class TestPaginationSlicing(unittest.TestCase):
    """paginate() returns correct slices."""

    def _make_pag(self, n=20):
        records = [{"id": i, "v": float(i)} for i in range(n)]
        pag = AggregatedPaginator(api_fetchers=[make_api(records)])
        pag.load_and_sort()
        return pag

    def test_first_page(self):
        pag = self._make_pag(20)
        result = pag.paginate(offset=0, limit=5)
        self.assertEqual(result["offset"], 0)
        self.assertEqual(result["limit"], 5)
        self.assertEqual(len(result["data"]), 5)

    def test_second_page(self):
        pag = self._make_pag(20)
        first = pag.paginate(offset=0, limit=5)
        second = pag.paginate(offset=5, limit=5)
        first_ids = {r["id"] for r in first["data"]}
        second_ids = {r["id"] for r in second["data"]}
        self.assertTrue(first_ids.isdisjoint(second_ids))

    def test_last_page_shorter_than_limit(self):
        pag = self._make_pag(12)
        result = pag.paginate(offset=10, limit=5)
        # Only 2 records remain after offset=10
        self.assertEqual(len(result["data"]), 2)

    def test_offset_beyond_total(self):
        pag = self._make_pag(5)
        result = pag.paginate(offset=100, limit=5)
        self.assertEqual(result["total"], 5)
        self.assertEqual(result["data"], [])

    def test_total_echoed(self):
        pag = self._make_pag(7)
        for offset in (0, 3, 6):
            result = pag.paginate(offset=offset, limit=2)
            self.assertEqual(result["total"], 7)

    def test_all_pages_cover_all_records(self):
        n = 17
        pag = self._make_pag(n)
        collected = []
        offset = 0
        limit = 5
        while True:
            page = pag.paginate(offset=offset, limit=limit)
            collected.extend(page["data"])
            if len(page["data"]) < limit:
                break
            offset += limit
        self.assertEqual(len(collected), n)

    def test_invalid_offset_raises(self):
        pag = self._make_pag()
        with self.assertRaises(ValueError):
            pag.paginate(offset=-1, limit=5)

    def test_invalid_limit_raises(self):
        pag = self._make_pag()
        with self.assertRaises(ValueError):
            pag.paginate(offset=0, limit=0)

    def test_paginate_before_load_raises(self):
        pag = AggregatedPaginator(api_fetchers=[make_api([])])
        with self.assertRaises(RuntimeError):
            pag.paginate(offset=0, limit=5)


class TestMLModelSorting(unittest.TestCase):
    """Sorting via pickle ML model works correctly."""

    def _save_model(self, model):
        """Save model to a temp pickle file and return the path."""
        tmp = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
        pickle.dump(model, tmp)
        tmp.close()
        return tmp.name

    def test_predict_model_orders_records(self):
        # Records from two "APIs"; scores deliberately reverse insertion order.
        api_a = [{"id": 1, "v": 1.0}, {"id": 2, "v": 2.0}]
        api_b = [{"id": 3, "v": 3.0}, {"id": 4, "v": 4.0}]
        # Score: id=4 → 0.9, id=3 → 0.7, id=2 → 0.5, id=1 → 0.1
        model = make_model([0.1, 0.5, 0.7, 0.9])
        model_path = self._save_model(model)

        try:
            pag = AggregatedPaginator(
                api_fetchers=[make_api(api_a), make_api(api_b)],
                model_path=model_path,
            )
            pag.load_and_sort(feature_keys=["v"])
            page = pag.paginate(offset=0, limit=4)
            ids = [r["id"] for r in page["data"]]
            # Descending by score: 4, 3, 2, 1
            self.assertEqual(ids, [4, 3, 2, 1])
        finally:
            os.unlink(model_path)

    def test_predict_proba_model_orders_records(self):
        records = [{"id": i, "v": float(i)} for i in range(1, 6)]
        # Scores: ascending i → ascending proba; highest proba = id 5
        scores = [0.1, 0.3, 0.5, 0.7, 0.9]
        model = make_proba_model(scores)
        model_path = self._save_model(model)

        try:
            pag = AggregatedPaginator(
                api_fetchers=[make_api(records)],
                model_path=model_path,
            )
            pag.load_and_sort(feature_keys=["v"])
            page = pag.paginate(offset=0, limit=5)
            ids = [r["id"] for r in page["data"]]
            self.assertEqual(ids, [5, 4, 3, 2, 1])
        finally:
            os.unlink(model_path)

    def test_ascending_sort(self):
        records = [{"id": i, "v": float(i)} for i in range(1, 4)]
        scores = [0.9, 0.5, 0.1]
        model = make_model(scores)
        model_path = self._save_model(model)

        try:
            pag = AggregatedPaginator(
                api_fetchers=[make_api(records)],
                model_path=model_path,
            )
            pag.load_and_sort(feature_keys=["v"], ascending=True)
            page = pag.paginate(offset=0, limit=3)
            ids = [r["id"] for r in page["data"]]
            # Ascending by score: id=3 (0.1) < id=2 (0.5) < id=1 (0.9)
            self.assertEqual(ids, [3, 2, 1])
        finally:
            os.unlink(model_path)

    def test_score_field_added_to_records(self):
        records = [{"id": 1, "v": 1.0}]
        model = make_model([0.42])
        model_path = self._save_model(model)

        try:
            pag = AggregatedPaginator(
                api_fetchers=[make_api(records)],
                model_path=model_path,
                score_field="ranking_score",
            )
            pag.load_and_sort(feature_keys=["v"])
            page = pag.paginate(offset=0, limit=1)
            self.assertAlmostEqual(page["data"][0]["ranking_score"], 0.42, places=5)
        finally:
            os.unlink(model_path)


class TestCustomSortKey(unittest.TestCase):
    """Custom sort_key takes precedence over ML model."""

    def test_custom_sort_key_used(self):
        records = [{"id": 1, "priority": 3},
                   {"id": 2, "priority": 1},
                   {"id": 3, "priority": 2}]
        pag = AggregatedPaginator(api_fetchers=[make_api(records)])
        pag.load_and_sort(sort_key=lambda r: r["priority"], ascending=True)
        page = pag.paginate(offset=0, limit=3)
        ids = [r["id"] for r in page["data"]]
        self.assertEqual(ids, [2, 3, 1])  # priority 1, 2, 3

    def test_custom_sort_key_descending(self):
        records = [{"id": 1, "score": 10},
                   {"id": 2, "score": 30},
                   {"id": 3, "score": 20}]
        pag = AggregatedPaginator(api_fetchers=[make_api(records)])
        pag.load_and_sort(sort_key=lambda r: r["score"], ascending=False)
        page = pag.paginate(offset=0, limit=3)
        ids = [r["id"] for r in page["data"]]
        self.assertEqual(ids, [2, 3, 1])  # scores 30, 20, 10

    def test_sort_key_overrides_model(self):
        """Even when a model is loaded, sort_key has priority."""
        records = [{"id": 1, "v": 1.0}, {"id": 2, "v": 2.0}]
        # Model would reverse order (score id=1=0.9, id=2=0.1)
        model = make_model([0.9, 0.1])
        tmp = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
        pickle.dump(model, tmp)
        tmp.close()
        model_path = tmp.name

        try:
            pag = AggregatedPaginator(
                api_fetchers=[make_api(records)],
                model_path=model_path,
            )
            # sort_key ignores model; sorts ascending by "v"
            pag.load_and_sort(
                feature_keys=["v"],
                sort_key=lambda r: r["v"],
                ascending=True,
            )
            page = pag.paginate(offset=0, limit=2)
            ids = [r["id"] for r in page["data"]]
            self.assertEqual(ids, [1, 2])
        finally:
            os.unlink(model_path)


class TestPaginationResponseShape(unittest.TestCase):
    """paginate() always returns the expected dict shape."""

    def test_response_keys(self):
        pag = AggregatedPaginator(api_fetchers=[make_api([{"id": 1}])])
        pag.load_and_sort()
        result = pag.paginate(offset=0, limit=5)
        self.assertIn("total", result)
        self.assertIn("offset", result)
        self.assertIn("limit", result)
        self.assertIn("data", result)

    def test_response_values_echoed(self):
        pag = AggregatedPaginator(api_fetchers=[make_api([{"id": i} for i in range(20)])])
        pag.load_and_sort()
        result = pag.paginate(offset=3, limit=7)
        self.assertEqual(result["offset"], 3)
        self.assertEqual(result["limit"], 7)
        self.assertEqual(result["total"], 20)
        self.assertEqual(len(result["data"]), 7)


if __name__ == "__main__":
    unittest.main()
