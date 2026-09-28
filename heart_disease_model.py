"""Heart disease model training and terminal screening logic."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Any

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "heart_disease.csv"


@dataclass(frozen=True)
class ModelConfig:
    """Immutable model parameters shared by the applications."""

    dataset_path: Path = DATASET_PATH
    test_size: float = 0.2
    random_state: int = 42
    svm_c: float = 1.0
    svm_kernel: str = "rbf"


CONFIG = ModelConfig()
_CACHE_LOCK = RLock()


def _normalize_binary(series: pd.Series) -> pd.Series:
    """Convert UCI boolean values into model-ready binary integers."""
    mapping = {True: 1, False: 0, "TRUE": 1, "FALSE": 0, "True": 1, "False": 0}
    converted = series.map(mapping)
    numeric = pd.to_numeric(series, errors="coerce")
    return converted.fillna(numeric).astype("Int64")


def calculate_health_age(row: pd.Series) -> float:
    """Calculate the coursework health-age feature."""
    age_delta = 0
    if pd.notna(row.get("trestbps")) and row["trestbps"] > 140:
        age_delta += 3
    if pd.notna(row.get("chol")) and row["chol"] > 240:
        age_delta += 4
    if pd.notna(row.get("fbs")) and row["fbs"] == 1:
        age_delta += 3
    if pd.notna(row.get("exang")) and row["exang"] == 1:
        age_delta += 3
    return row["age"] + age_delta


def load_dataset(path: Path) -> pd.DataFrame:
    """Load and normalize the local UCI dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_csv(path).rename(columns={"thalch": "thalach"})
    frame = frame.drop(columns=[column for column in ("id", "dataset") if column in frame])
    for column in ("fbs", "exang"):
        if column in frame:
            frame[column] = _normalize_binary(frame[column])
    if "num" not in frame:
        raise ValueError("Dataset must contain the UCI target column 'num'.")
    frame["target"] = (pd.to_numeric(frame.pop("num"), errors="raise") > 0).astype(int)
    frame["Health_Age"] = frame.apply(calculate_health_age, axis=1)
    return frame


@dataclass(frozen=True)
class ModelBundle:
    """Trained pipeline and metadata consumed by both interfaces."""

    pipeline: Pipeline
    feature_columns: tuple[str, ...]
    categorical_options: dict[str, tuple[Any, ...]]
    metrics: dict[str, Any]


def train_model(config: ModelConfig) -> ModelBundle:
    """Train and evaluate the calibrated SVM pipeline."""
    frame = load_dataset(config.dataset_path)
    features = frame.drop(columns="target")
    target = frame["target"]
    numeric_features = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [
        column for column in features.columns
        if pd.api.types.is_object_dtype(features[column])
        or pd.api.types.is_string_dtype(features[column])
        or isinstance(features[column].dtype, pd.CategoricalDtype)
    ]
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])
    classifier = CalibratedClassifierCV(
        estimator=SVC(
            kernel=config.svm_kernel,
            C=config.svm_c,
            gamma="scale",
            class_weight="balanced",
            random_state=config.random_state,
        ),
        cv=5,
        ensemble=False,
    )
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=config.test_size,
        random_state=config.random_state, stratify=target,
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    options = {
        column: tuple(sorted(features[column].dropna().unique().tolist(), key=str))
        for column in categorical_features
    }
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "classification_report": classification_report(y_test, predictions, output_dict=True),
    }
    return ModelBundle(pipeline, tuple(features.columns), options, metrics)


@lru_cache(maxsize=4)
def _cached_model(config: ModelConfig) -> ModelBundle:
    """Cache model training behind a lock for safe concurrent access."""
    with _CACHE_LOCK:
        return train_model(config)


def get_model() -> ModelBundle:
    """Return the shared trained model bundle."""
    return _cached_model(CONFIG)


def predict_patient(patient: dict[str, Any]) -> dict[str, Any]:
    """Score a structured patient record using the shared model."""
    bundle = get_model()
    frame = pd.DataFrame([patient])
    frame["Health_Age"] = frame.apply(calculate_health_age, axis=1)
    frame = frame.reindex(columns=bundle.feature_columns)
    return {
        "prediction": int(bundle.pipeline.predict(frame)[0]),
        "probability": float(bundle.pipeline.predict_proba(frame)[0, 1]),
        "health_age": int(frame["Health_Age"].iloc[0]),
    }


def _number(label: str, minimum: float, maximum: float, default: float) -> float:
    """Read a bounded numeric terminal value."""
    while True:
        raw = input(f"{label} [{default}]: ").strip()
        if not raw:
            return default
        try:
            value = float(raw)
        except ValueError:
            print("Enter a numeric value.")
            continue
        if minimum <= value <= maximum:
            return value
        print(f"Value must be between {minimum} and {maximum}.")


def _choice(label: str, options: tuple[Any, ...]) -> Any:
    """Read a categorical value by menu index."""
    print(f"\n{label}")
    for index, option in enumerate(options, start=1):
        print(f"  {index}. {option}")
    while True:
        try:
            return options[int(input("Selection number: ").strip()) - 1]
        except (ValueError, IndexError):
            print(f"Choose a number from 1 to {len(options)}.")


def collect_patient() -> dict[str, Any]:
    """Collect a structured patient record for console screening."""
    options = get_model().categorical_options
    return {
        "age": _number("Age in years", 18, 120, 50),
        "sex": _choice("Sex", options["sex"]),
        "cp": _choice("Chest pain type", options["cp"]),
        "trestbps": _number("Resting blood pressure (mm Hg)", 60, 260, 130),
        "chol": _number("Serum cholesterol (mg/dl)", 80, 700, 210),
        "fbs": int(_choice("Fasting blood sugar > 120 mg/dl", (0, 1))),
        "restecg": _choice("Resting ECG", options["restecg"]),
        "thalach": _number("Maximum heart rate", 40, 250, 150),
        "exang": int(_choice("Exercise-induced angina", (0, 1))),
        "oldpeak": _number("ST depression", 0, 10, 1.0),
        "slope": _choice("Exercise ST slope", options["slope"]),
        "ca": _number("Major vessels (0-3)", 0, 3, 0),
        "thal": _choice("Thalassemia status", options["thal"]),
    }


def run_console() -> None:
    """Run the terminal application."""
    print("\nUCI Heart Disease Clinical Screening")
    print("Values are for academic screening only and are not a diagnosis.\n")
    result = predict_patient(collect_patient())
    print("\n" + "=" * 52)
    print("CLINICAL SCREENING RESULT")
    print("=" * 52)
    print(f"Calculated health age : {result['health_age']} years")
    print(f"Estimated risk        : {result['probability']:.1%}")
    print(f"Risk classification   : {'HIGH RISK' if result['prediction'] else 'LOW RISK'}")
    print("This result supports coursework demonstration, not clinical care.")
    print("=" * 52)


if __name__ == "__main__":
    run_console()
