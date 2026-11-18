import numpy as np

from scripts.analyze_corrected_protein_critic_xgboost import class_diagnostics


def test_class_diagnostics_preserve_vocab_order_and_support():
    names = ["PF00001", "PF00002", "PF00003"]
    y_true = np.asarray([0, 0, 1, 2, 2, 2])
    prediction = np.asarray([0, 1, 1, 0, 2, 2])

    result = class_diagnostics(y_true, prediction, names)

    assert [entry["class_name"] for entry in result["classes"]] == names
    assert [entry["support"] for entry in result["classes"]] == [2, 1, 3]
    assert [entry["recall"] for entry in result["classes"]] == [0.5, 1.0, 2 / 3]
    matrix = result["confusion_matrix"]["rows_are_true_columns_are_predicted"]
    assert matrix == [[1, 1, 0], [0, 1, 0], [1, 0, 2]]


def test_class_diagnostics_reports_zero_support_class_explicitly():
    result = class_diagnostics(np.asarray([0, 0]), np.asarray([0, 1]), ["A", "B", "C"])

    assert result["classes"][2]["support"] == 0
    assert result["classes"][2]["recall"] == 0.0
    assert result["confusion_matrix"]["rows_are_true_columns_are_predicted"][0] == [1, 1, 0]
