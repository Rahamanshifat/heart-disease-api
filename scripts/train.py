"""Train a Random Forest on the Heart Disease dataset and save it with joblib."""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]


def main(csv_path: Path = ROOT / "data" / "heart.csv") -> None:
    df = pd.read_csv(csv_path).drop_duplicates()
    X, y = df[FEATURES], df["target"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, model.predict(X_te))
    print(f"rows={len(df)} test accuracy={acc:.3f}")

    out = ROOT / "model"
    out.mkdir(exist_ok=True)
    joblib.dump(model, out / "heart_model.joblib")
    (out / "metadata.json").write_text(json.dumps(
        {"model_type": "RandomForestClassifier", "features": FEATURES,
         "test_accuracy": round(float(acc), 4)}, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "heart.csv")
