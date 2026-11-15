"""Run directly, or use unittest discovery from the repository root."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent
MODULE = HERE / "audit_oof.py"
if not MODULE.exists():
    MODULE = HERE.parent / "tools" / "audit_oof.py"
spec = importlib.util.spec_from_file_location("oof_audit_under_test", MODULE)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fixture():
    rows = []
    for group, fold in [("exp_a", 0), ("exp_b", 1)]:
        for v in audit.VERSION_WEIGHTS:
            step = {"k1o0": 1, "k2o0": 2, "k3o0": 3}[v]
            for i in range(0, 36, step):
                label = i // 6 + 1
                row = {"sample_id": f"{group}_{i}", "experiment_id": group + "__" + v,
                       "grp": group, "ver": v, "fold": fold, "y": label,
                       "relative_time_s": i * 5}
                row.update({f"p{k}": float(k == label) for k in audit.LABELS})
                rows.append(row)
    return pd.DataFrame(rows)


class AuditTests(unittest.TestCase):
    def assert_bad(self, df, text, manifest=None):
        result = audit.audit_oof(df, manifest)
        self.assertFalse(result["ok"])
        self.assertTrue(any(text in s for s in result["errors"]), result["errors"])

    def test_valid_grouped_oof(self):
        result = audit.audit_oof(fixture())
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["experiments"], 2)
        self.assertEqual(result["raw_oof_development_scores"]["p"]["weighted_macro_f1"], 1)

    def test_original_group_can_be_recovered_from_version_suffix(self):
        self.assertTrue(audit.audit_oof(fixture().drop(columns="grp"))["ok"])

    def test_augmentation_fold_leak_is_detected_without_grp(self):
        df = fixture().drop(columns="grp")
        df.loc[(df.ver == "k2o0") & df.experiment_id.str.startswith("exp_a"), "fold"] = 1
        self.assert_bad(df, "span multiple folds")

    def test_duplicate_sample_version(self):
        df = fixture()
        self.assert_bad(pd.concat([df, df.iloc[:1]], ignore_index=True), "Duplicate (sample_id, ver)")

    def test_label_disagreement_across_versions(self):
        df = fixture()
        df.loc[(df.sample_id == "exp_a_0") & (df.ver == "k2o0"), "y"] = 2
        self.assert_bad(df, "inconsistent labels")

    def test_nonfinite_probability(self):
        df = fixture()
        df.loc[0, "p1"] = np.nan
        self.assert_bad(df, "Invalid probability")

    def test_not_normalized_probability(self):
        df = fixture()
        df.loc[0, "p1"] = .5
        self.assert_bad(df, "sum to 1")

    def test_missing_class_probability(self):
        self.assert_bad(fixture().drop(columns="p6"), "Incomplete probability block")

    def test_missing_version_not_silently_reweighted(self):
        df = fixture()
        self.assert_bad(df[df.ver != "k3o0"], "all three versions")
        with self.assertRaises(ValueError):
            audit.weighted_macro_f1(df[df.ver != "k3o0"], df.loc[df.ver != "k3o0", "y"])

    def test_manifest_disagreement(self):
        manifest = pd.DataFrame({"grp": ["exp_a", "exp_b"], "fold": [1, 0]})
        self.assert_bad(fixture(), "disagrees with manifest", manifest)

    def test_manifest_match(self):
        manifest = pd.DataFrame({"grp": ["exp_a", "exp_b"], "fold": [0, 1]})
        self.assertTrue(audit.audit_oof(fixture(), manifest)["ok"])

    def test_version_coverage_mismatch(self):
        df = fixture()
        self.assert_bad(df[~((df.grp == "exp_b") & (df.ver == "k3o0"))], "different original experiments")

    def test_probability_class_order_is_numeric(self):
        df = fixture()
        cols = [c for c in df if not c.startswith("p")] + [f"p{k}" for k in reversed(audit.LABELS)]
        self.assertTrue(audit.audit_oof(df[cols])["ok"])
        self.assertEqual(audit.audit_oof(df[cols])["raw_oof_development_scores"]["p"]["weighted_macro_f1"], 1)

    def test_weighted_objective_is_not_pooled_macro(self):
        df = fixture()
        pred = df.y.to_numpy().copy()
        pred[df.ver.ne("k1o0").to_numpy()] = 1
        score, versions = audit.weighted_macro_f1(df, pred)
        expected = (9 * versions["k1o0"] + versions["k2o0"] + versions["k3o0"]) / 11
        self.assertAlmostEqual(score, expected)
        pooled = f1_score(df.y, pred, labels=audit.LABELS, average="macro", zero_division=0)
        self.assertGreater(abs(score - pooled), .01)

    def test_duplicate_timestamp(self):
        df = fixture()
        df.loc[1, "relative_time_s"] = df.loc[0, "relative_time_s"]
        self.assert_bad(df, "Duplicate timestamp")


if __name__ == "__main__":
    unittest.main()
