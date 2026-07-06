"""
Train Machine Learning Models for Smart Lender
----------------------------------------------
This script performs:

1. Load Dataset
2. Data Cleaning
3. Feature Engineering
4. Preprocessing
5. Model Training
6. Model Evaluation
7. Save Best Model
"""

from pathlib import Path
import json
import logging

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# Logging


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s : %(message)s"
)

logger = logging.getLogger(__name__)


# Paths


BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR.parent / "data" / "loan_approval_dataset.csv"

MODEL_PATH = BASE_DIR / "best_model.pkl"

PREPROCESSOR_PATH = BASE_DIR / "preprocessor.pkl"

METRICS_PATH = BASE_DIR / "metrics.json"


# Load Dataset



def load_dataset(path: Path) -> pd.DataFrame:
    """
    Load dataset.
    """

    logger.info("Loading Dataset...")

    df = pd.read_csv(path)

    # Remove spaces from column names
    df.columns = df.columns.str.strip()

    # Remove spaces from string values
    for column in df.select_dtypes(include=["object", "string"]):
        df[column] = df[column].str.strip()

    logger.info("Dataset Loaded Successfully.")

    return df



# Data Cleaning



def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataset.
    """

    logger.info("Cleaning Dataset...")

    # Remove duplicates
    df = df.drop_duplicates()

    # Remove missing values
    df = df.dropna()

    logger.info("Dataset Cleaned Successfully.")

    return df



# Feature Engineering



def prepare_features(df: pd.DataFrame):
    """
    Prepare X and y.
    """

    # Remove Loan ID
    df = df.drop(columns=["loan_id"])

    X = df[
    [
        "no_of_dependents",
        "education",
        "self_employed",
        "income_annum",
        "loan_amount",
        "loan_term",
        "cibil_score",
    ]
]

    y = df["loan_status"].map({
    "Rejected": 0,
    "Approved": 1
})

    return X, y



# Train Test Split



def split_dataset(X, y):

    logger.info("Splitting Dataset...")

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )


# Preprocessing


def build_preprocessor():
    """
    Build preprocessing pipeline.
    """

    logger.info("Building Preprocessor...")

    categorical_features = [
        "education",
        "self_employed"
    ]

    numerical_features = [
    "no_of_dependents",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features,
            ),
            (
                "numerical",
                StandardScaler(),
                numerical_features,
            ),
        ]
    )

    logger.info("Preprocessor Created Successfully.")

    return preprocessor



# Machine Learning Models


def build_models():
    """
    Create all machine learning models.
    """

    logger.info("Building Models...")

    models = {
        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),

        "KNN": KNeighborsClassifier(
            n_neighbors=5
        ),

        "XGBoost": XGBClassifier(
            random_state=42,
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            eval_metric="logloss",
        ),
    }

    logger.info("Models Created Successfully.")

    return models



# Build Pipeline


def build_pipeline(
    preprocessor,
    model,
):
    """
    Create complete ML pipeline.
    """

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    return pipeline



# Train Models


def train_models(
    models: dict,
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
):
    """
    Train all models and return the best model.
    """

    logger.info("=" * 60)
    logger.info("Training Models...")
    logger.info("=" * 60)

    best_model = None
    best_pipeline = None
    best_accuracy = 0.0

    metrics = {}

    for name, model in models.items():

        logger.info(f"Training {name}...")

        pipeline = build_pipeline(
            preprocessor,
            model,
        )

        pipeline.fit(
            X_train,
            y_train,
        )

        predictions = pipeline.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            predictions,
        )

        precision = precision_score(
            y_test,
            predictions,
            pos_label=1,
        )

        recall = recall_score(
            y_test,
            predictions,
            pos_label=1,
        )

        f1 = f1_score(
            y_test,
            predictions,
            pos_label=1,
        )

        metrics[name] = {
            "Accuracy": round(accuracy, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1 Score": round(f1, 4),
        }

        logger.info(f"{name}")
        logger.info(f"Accuracy : {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall   : {recall:.4f}")
        logger.info(f"F1 Score : {f1:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = name
            best_pipeline = pipeline

    logger.info("=" * 60)
    logger.info(f"Best Model : {best_model}")
    logger.info(f"Accuracy   : {best_accuracy:.4f}")
    logger.info("=" * 60)

    return (
        best_pipeline,
        metrics,
    )


# Save Artifacts


def save_artifacts(
    pipeline: Pipeline,
    metrics: dict,
) -> None:
    """
    Save trained model and evaluation metrics.
    """

    logger.info("Saving model...")

    joblib.dump(
        pipeline,
        MODEL_PATH,
    )

    preprocessor = pipeline.named_steps["preprocessor"]

    joblib.dump(
        preprocessor,
        PREPROCESSOR_PATH,
    )

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )

    logger.info("Artifacts saved successfully.")



# Main Function


def main() -> None:
    """
    Complete training pipeline.
    """

    logger.info("=" * 60)
    logger.info("SMART LENDER MODEL TRAINING")
    logger.info("=" * 60)

    # Load dataset
    df = load_dataset(DATASET_PATH)

    # Clean dataset
    df = clean_dataset(df)

    # Features
    X, y = prepare_features(df)

    # Split dataset
    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_dataset(
        X,
        y,
    )

    # Preprocessor
    preprocessor = build_preprocessor()

    # Models
    models = build_models()

    # Train models
    best_pipeline, metrics = train_models(
        models=models,
        preprocessor=preprocessor,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    # Save artifacts
    save_artifacts(
        pipeline=best_pipeline,
        metrics=metrics,
    )

    logger.info("=" * 60)
    logger.info("Training Completed Successfully")
    logger.info("=" * 60)



# Entry Point


if __name__ == "__main__":
    main()