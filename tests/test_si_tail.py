import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from si_tail import (RawLogEnsemble, TanimotoRegressor, fit_outer_fold,
                     inverse_log, masked_mean_pool, scaffold_folds,
                     score_predictions, smiles_features, training_weights)


class RecordingTransform(TransformerMixin, BaseEstimator):
    fits = []

    def fit(self, X, y=None):
        self.__class__.fits.append(np.array(X[:, 0], copy=True))
        self.mean_ = np.mean(X, axis=0)
        return self

    def transform(self, X):
        return X - self.mean_


class TailTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(51)
        self.X = rng.normal(size=(36, 6))
        self.groups = np.repeat(np.arange(12), 3)
        self.y = np.exp(1 + self.X[:, 0])
        self.y[0] = 36066.667

    def test_mixture_reaches_convex_mse_optimum(self):
        model = RawLogEnsemble(Ridge(alpha=10)).fit(self.X, self.y, self.groups)
        raw, log = model.inner_oof_.T
        difference = raw - log
        optimum = np.clip(np.dot(difference, self.y - log) / np.dot(difference, difference), 0, 1)
        self.assertAlmostEqual(model.weights_[0], optimum, places=8)
        self.assertAlmostEqual(model.weights_.sum(), 1)
        self.assertGreaterEqual(model.weights_.min(), 0)
        np.testing.assert_allclose(model.predict(self.X), model.predict_components(self.X) @ model.weights_)
        clone(model).fit(self.X, self.y, self.groups)

    def test_preprocessing_fits_only_current_training_rows(self):
        X = self.X.copy()
        X[:, 0] = np.arange(len(X))
        train, valid = next(iter(scaffold_folds(self.groups, 3)))
        RecordingTransform.fits.clear()
        estimator = make_pipeline(RecordingTransform(), Ridge(alpha=100))
        model = RawLogEnsemble(estimator).fit(X[train], self.y[train], self.groups[train])
        self.assertEqual(len(RecordingTransform.fits), 8)  # 2 x (3 inner + full refit)
        for rows in RecordingTransform.fits:
            self.assertTrue(set(rows).issubset(X[train, 0]))
            self.assertTrue(set(rows).isdisjoint(X[valid, 0]))
        for fold in np.unique(model.inner_fold_):
            held = model.inner_fold_ == fold
            self.assertTrue(set(self.groups[train][held]).isdisjoint(self.groups[train][~held]))

    def test_outer_validation_labels_do_not_change_predictions_or_selection(self):
        # Small tree makes this an integration test of selection, not a LightGBM speed test.
        X = (self.X > 0).astype(float)
        features = {"fingerprints": X, "descriptors": self.X[:, :2]}
        train, valid = next(iter(scaffold_folds(self.groups, 3)))

        def run(labels):
            return fit_outer_fold({k: v[train] for k, v in features.items()}, labels[train],
                                  self.groups[train], {k: v[valid] for k, v in features.items()})

        with patch("si_tail.boosting", side_effect=lambda seed: DecisionTreeRegressor(max_depth=2, random_state=seed)):
            first, first_choices, _ = run(self.y)
            poisoned = self.y.copy()
            poisoned[valid] = 1e15
            second, second_choices, _ = run(poisoned)
        self.assertEqual(first_choices, second_choices)
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])

    def test_weighting_is_mild_normalized_and_training_only(self):
        for mode in ["uniform", "log", "raw"]:
            weights = training_weights(self.y, mode, 0.5)
            self.assertTrue(np.isfinite(weights).all())
            self.assertGreater(weights.min(), 0)
            self.assertAlmostEqual(weights.mean(), 1)
            if mode != "uniform":
                self.assertGreater(weights[self.y.argmax()], weights[self.y.argmin()])
        np.testing.assert_array_equal(training_weights(np.zeros(5), "log", 1), np.ones(5))
        with self.assertRaises(ValueError):
            training_weights(self.y, "raw", -1)

    def test_metrics_are_unweighted_and_do_not_clip_predictions(self):
        y, pred = np.array([0., 10., 36000.]), np.array([-5., 2000., 50000.])
        scores = score_predictions(y, pred)
        self.assertEqual(scores["MSE"], np.mean((y - pred) ** 2))
        self.assertEqual(scores["negative_predictions"], 1)
        self.assertEqual(scores["pred_max"], 50000)
        self.assertEqual(scores["low_pred_gt_1000"], 1)
        self.assertGreater(inverse_log(np.array([20.]))[0], 1e8)
        with self.assertRaises(FloatingPointError):
            inverse_log(np.array([1000.]))
        with self.assertRaises(ValueError):
            score_predictions(y, pred[:, None])

    def test_equivalent_molecules_and_scaffolds_cannot_cross_folds(self):
        smiles = ["CC[NH3+].[Cl-]", "CCN", "c1ccccc1O", "c1ccccc1C",
                  "C[C@H]1CCCCO1", "C[C@@H]1CCCCO1", "C1CCC1", "C1CCCC1"]
        features, keys, groups = smiles_features(smiles)
        self.assertEqual(keys[0], keys[1])
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(groups[2], groups[3])
        self.assertEqual(groups[4], groups[5])
        for train, valid in scaffold_folds(groups):
            self.assertTrue(set(np.array(keys)[train]).isdisjoint(np.array(keys)[valid]))
        self.assertEqual(features["fingerprints"].shape, (8, 2048))

    def test_frozen_pooling_ignores_special_and_padding_tokens(self):
        import torch
        hidden = torch.tensor([[[999., 999.], [1., 3.], [3., 5.], [-999., -999.], [1e6, 1e6]]])
        attention = torch.tensor([[1, 1, 1, 1, 0]])
        special = torch.tensor([[1, 0, 0, 1, 1]])
        np.testing.assert_allclose(masked_mean_pool(hidden, attention, special).numpy(), [[2, 4]])

    def test_descriptor_transform_is_stateless_and_leaves_bits_and_groups(self):
        smiles = ["CCN", "c1ccccc1O", "C1CCCC1"]
        raw, raw_keys, raw_groups = smiles_features(smiles)
        transformed, keys, groups = smiles_features(smiles, "signedlog")
        np.testing.assert_array_equal(raw_groups, groups)
        self.assertEqual(raw_keys, keys)
        np.testing.assert_array_equal(raw["fingerprints"], transformed["fingerprints"])
        np.testing.assert_allclose(transformed["descriptors"],
                                   np.sign(raw["descriptors"]) * np.log1p(np.abs(raw["descriptors"])))

    def test_tanimoto_estimator_is_cloneable_and_unclipped(self):
        X = (self.X > 0).astype(float)
        model = clone(TanimotoRegressor()).fit(X, self.y)
        self.assertEqual(model.predict(X).shape, self.y.shape)
        with self.assertRaises(ValueError):
            list(scaffold_folds(np.zeros(5)))


if __name__ == "__main__":
    unittest.main()
