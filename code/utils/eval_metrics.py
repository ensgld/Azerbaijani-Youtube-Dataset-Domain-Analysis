"""Macro-F1 evaluation helpers (required)."""

import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix

def predict_labels(y_prob, num_classes=3):
    y_prob = np.array(y_prob)
    return y_prob.argmax(axis=1)

def evaluate_macro_f1(model, X_test, y_test, num_classes=3, batch_size=256, title=""):
    y_prob = model.predict(X_test, batch_size=batch_size, verbose=0)
    y_pred = predict_labels(y_prob, num_classes=num_classes)

    macro = f1_score(y_test, y_pred, average="macro")
    print(f"\n{title} Macro-F1: {macro:.4f}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    return macro, y_pred
