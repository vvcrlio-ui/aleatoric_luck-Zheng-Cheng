from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from nlsy_replication.src.SHAP_experiment import evaluate_feature_count
from nlsy_replication.src.evaluation import (
    r2_against_training_mean,
    training_mean_null_mse,
)
from nlsy_replication.src.experiment import (
    add_metadata,
    build_experiment_metadata,
    load_checkpoint,
    parallel_preference,
    write_checkpoint,
)
from nlsy_replication.src.feature_sets import main as feature_sets_main
from nlsy_replication.src.model_registry import MODEL_NAMES, make_model
from nlsy_replication.src.sample_size import fit_power_law


class NLSYReplicationTests(unittest.TestCase):
    def test_linear_registry_model_fits(self):
        rng = np.random.default_rng(7)
        X = pd.DataFrame(rng.normal(size=(60, 4)), columns=list("abcd"))
        y = 0.5 * X["a"] - 0.2 * X["b"] + rng.normal(0, 0.05, len(X))
        model = make_model("ridge", seed=12345, n_jobs=1)
        model.fit(X, y)
        self.assertEqual(model.predict(X).shape, (60,))
        self.assertIn("lightgbm", MODEL_NAMES)
        self.assertIn("bart", MODEL_NAMES)

    def test_bart_defaults_match_paper_and_use_process_parallelism(self):
        model = make_model("bart", seed=12345, n_jobs=1)
        self.assertEqual(model.n_trees, 200)
        self.assertEqual(model.n_samples, 1000)
        self.assertEqual(model.n_burn, 100)
        self.assertEqual(model.thin, 1.0)
        self.assertEqual(parallel_preference(["ridge", "bart"]), "processes")
        self.assertEqual(parallel_preference(["ridge"]), "threads")

    def test_training_mean_metric_uses_supplied_subset(self):
        y_test = np.array([0.0, 10.0])
        y_subset = np.array([0.0, 0.0])
        self.assertEqual(training_mean_null_mse(y_test, y_subset), 50.0)
        self.assertEqual(r2_against_training_mean(25.0, y_test, y_subset), 0.5)

    def test_experiment_identity_and_checkpoint_are_scoped(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data = root / "data.csv"
            output = root / "results.csv"
            data.write_text("Aset1_x,outcome_a,outcome_b\n1,2,3\n")
            common = {
                "kind": "feature_sets",
                "data_path": data,
                "test_size": 0.3,
                "split_seed": 12345,
                "extra": {"n_sizes": 2},
            }
            first = build_experiment_metadata(outcome="outcome_a", **common)
            second = build_experiment_metadata(outcome="outcome_b", **common)
            self.assertNotEqual(first["experiment_id"], second["experiment_id"])

            row = {"model": "ridge", "k": 1, "seed": 12345, "status": "ok"}
            write_checkpoint(
                pd.DataFrame(),
                [add_metadata(row, first), add_metadata(row, second)],
                output,
                key_columns=["model", "k", "seed"],
                sort_columns=["model", "k", "seed"],
            )
            saved = load_checkpoint(output)
            self.assertEqual(len(saved), 2)
            self.assertEqual(set(saved["outcome"]), {"outcome_a", "outcome_b"})

    def test_legacy_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "legacy.csv"
            pd.DataFrame([{"model": "ridge", "k": 1}]).to_csv(output, index=False)
            with self.assertRaisesRegex(ValueError, "lacks experiment metadata"):
                load_checkpoint(output)

    def test_feature_cli_does_not_resume_across_outcomes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_path = root / "data.csv"
            output = root / "feature_results.csv"
            rng = np.random.default_rng(19)
            frame = pd.DataFrame(
                rng.normal(size=(60, 4)),
                columns=["Aset1_a", "Aset2_b", "Bset1_c", "Bset2_d"],
            )
            frame["outcome_a"] = frame["Aset1_a"] + rng.normal(size=len(frame))
            frame["outcome_b"] = frame["Bset1_c"] + rng.normal(size=len(frame))
            frame.to_csv(data_path, index=False)

            for outcome in ("outcome_a", "outcome_b"):
                argv = [
                    "feature_sets.py",
                    "--data",
                    str(data_path),
                    "--outcome",
                    outcome,
                    "--out",
                    str(output),
                    "--models",
                    "ols",
                    "--n-sizes",
                    "2",
                    "--n-draws",
                    "1",
                    "--batch-size",
                    "2",
                    "--n-jobs",
                    "1",
                ]
                with patch.object(sys, "argv", argv):
                    feature_sets_main()

            saved = pd.read_csv(output)
            self.assertEqual(saved["experiment_id"].nunique(), 2)
            self.assertEqual(set(saved["outcome"]), {"outcome_a", "outcome_b"})

    def test_shap_ordering_returns_both_directions(self):
        rng = np.random.default_rng(11)
        X = pd.DataFrame(rng.normal(size=(60, 3)), columns=["f1", "f2", "f3"])
        y = pd.Series(X["f1"] + rng.normal(0, 0.1, len(X)))
        rows = evaluate_feature_count(
            1,
            ["f1", "f2", "f3"],
            X.iloc[:45],
            X.iloc[45:],
            y.iloc[:45],
            y.iloc[45:],
            seed=12345,
        )
        self.assertEqual({row["direction"] for row in rows}, {"high_to_low", "low_to_high"})
        self.assertEqual(len(rows), 2)

    def test_power_law_fit_is_guarded_per_model(self):
        rows = []
        for n in [100, 200, 400, 800, 1600]:
            for draw in range(6):
                rows.append(
                    {
                        "model": "known",
                        "n_samples": n,
                        "mse": 1.5 * n ** -0.6 + 0.03 + (draw - 2.5) * 1e-5,
                        "status": "ok",
                    }
                )
        for n in [100, 200, 400]:
            rows.append(
                {"model": "too_few", "n_samples": n, "mse": 0.1, "status": "ok"}
            )
        fits = fit_power_law(pd.DataFrame(rows), bootstrap_iterations=20, seed=7)
        known = fits[fits["model"].eq("known")].iloc[0]
        failed = fits[fits["model"].eq("too_few")].iloc[0]
        self.assertAlmostEqual(float(known["epsilon"]), 0.03, places=3)
        self.assertIn(known["status"], {"stable", "unstable"})
        self.assertEqual(failed["status"], "fit_failed")

    def test_colab_notebook_is_valid_json_without_saved_outputs(self):
        path = Path(__file__).parents[1] / "nlsy_replication" / "colab_run.ipynb"
        notebook = json.loads(path.read_text())
        self.assertEqual(notebook["nbformat"], 4)
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertTrue(code_cells)
        self.assertTrue(all(cell.get("outputs") == [] for cell in code_cells))


if __name__ == "__main__":
    unittest.main()
