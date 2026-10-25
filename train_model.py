import argparse
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE = Path(__file__).resolve().parent
INDIA_CSV = BASE / "data" / "india_accidents.csv"
US_CSV = BASE / "data" / "US_Accidents_March23.csv"
DEMO_CSV = BASE / "data" / "demo_accidents.csv"
DEFAULT_CSV = INDIA_CSV if INDIA_CSV.exists() else (US_CSV if US_CSV.exists() else DEMO_CSV)
MODEL_PATH = BASE / "models" / "severity_model.joblib"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="Path to accident CSV")
    parser.add_argument("--max-rows", type=int, default=150000)
    args = parser.parse_args()
    path = Path(args.csv)
    if not path.exists():
        raise SystemExit(f"CSV not found: {path}")

    wanted = ["Severity", "Start_Time", "Start_Lat", "Start_Lng",
              "Temperature(F)", "Humidity(%)", "Visibility(mi)", "Weather_Condition"]
    df = pd.read_csv(path, usecols=lambda c: c in wanted, nrows=args.max_rows, low_memory=False)
    missing = [c for c in wanted if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns: {missing}. Check your CSV column names.")

    df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")
    df = df.dropna(subset=["Severity", "Start_Time", "Start_Lat", "Start_Lng"])
    df["hour"] = df["Start_Time"].dt.hour
    df["month"] = df["Start_Time"].dt.month
    df["Weather_Condition"] = df["Weather_Condition"].fillna("Unknown").astype(str)
    numeric = ["Start_Lat", "Start_Lng", "Temperature(F)", "Humidity(%)", "Visibility(mi)", "hour", "month"]
    categorical = ["Weather_Condition"]
    X = df[numeric + categorical]
    y = df["Severity"].astype(int)

    if y.nunique() < 2:
        raise SystemExit("Need at least two severity classes to train a classifier.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical)
    ])
    model = Pipeline([
        ("preprocess", pre),
        ("classifier", RandomForestClassifier(
            n_estimators=100, min_samples_leaf=2, class_weight="balanced_subsample",
            random_state=42, n_jobs=1
        ))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"Rows used: {len(df):,}")
    print(f"Accuracy: {accuracy_score(y_test, pred):.3f}")
    print(f"Weighted F1: {f1_score(y_test, pred, average='weighted'):.3f}")
    print("\\nPer-class report:")
    print(classification_report(y_test, pred, zero_division=0))
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\\nSaved model: {MODEL_PATH}")

if __name__ == "__main__":
    main()
