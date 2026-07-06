"""
Exploratory Data Analysis (EDA)
--------------------------------
This script performs:
1. Data Loading
2. Dataset Inspection
3. Data Cleaning
4. Summary Statistics
5. Visualization
"""

from pathlib import Path
import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ==========================================================
# Logging Configuration
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s : %(message)s"
)

logger = logging.getLogger(__name__)

# ==========================================================
# Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR.parent / "data" / "loan_approval_dataset.csv"

PLOTS_DIR = BASE_DIR / "plots"

PLOTS_DIR.mkdir(exist_ok=True)

# ==========================================================
# Plot Style
# ==========================================================

plt.style.use("ggplot")
sns.set_theme(style="whitegrid")

# ==========================================================
# Load Dataset
# ==========================================================


def load_dataset(path: Path) -> pd.DataFrame:
    """
    Load dataset from CSV.

    Parameters
    ----------
    path : Path
        CSV file path.

    Returns
    -------
    pd.DataFrame
    """

    logger.info("Loading dataset...")

    df = pd.read_csv(path)

    # Remove leading/trailing spaces
    df.columns = df.columns.str.strip()

    logger.info("Dataset loaded successfully.")

    return df


# ==========================================================
# Basic Information
# ==========================================================


def dataset_overview(df: pd.DataFrame) -> None:
    """
    Display dataset overview.
    """

    print("\n" + "=" * 70)
    print("DATASET OVERVIEW")
    print("=" * 70)

    print("\nShape")
    print(df.shape)

    print("\nColumns")
    print(df.columns.tolist())

    print("\nFirst Five Rows")
    print(df.head())

    print("\nLast Five Rows")
    print(df.tail())


# ==========================================================
# Data Types
# ==========================================================


def data_types(df: pd.DataFrame) -> None:
    """
    Print data types.
    """

    print("\n" + "=" * 70)
    print("DATA TYPES")
    print("=" * 70)

    print(df.dtypes)


# ==========================================================
# Missing Values
# ==========================================================


def missing_values(df: pd.DataFrame) -> None:
    """
    Display missing values.
    """

    print("\n" + "=" * 70)
    print("MISSING VALUES")
    print("=" * 70)

    print(df.isnull().sum())


# ==========================================================
# Duplicate Values
# ==========================================================


def duplicate_values(df: pd.DataFrame) -> None:
    """
    Display duplicate rows.
    """

    print("\n" + "=" * 70)
    print("DUPLICATE VALUES")
    print("=" * 70)

    duplicates = df.duplicated().sum()

    print(f"Duplicate Rows : {duplicates}")


# ==========================================================
# Statistical Summary
# ==========================================================


def statistical_summary(df: pd.DataFrame) -> None:
    """
    Display statistical summary.
    """

    print("\n" + "=" * 70)
    print("STATISTICAL SUMMARY")
    print("=" * 70)

    print(df.describe())


# ==========================================================
# Unique Values
# ==========================================================


def unique_values(df: pd.DataFrame) -> None:
    """
    Print unique values in each column.
    """

    print("\n" + "=" * 70)
    print("UNIQUE VALUES")
    print("=" * 70)

    for column in df.columns:
        print(f"\n{column}")
        print(df[column].unique())


# ==========================================================
# Value Counts
# ==========================================================


def value_counts(df: pd.DataFrame) -> None:
    """
    Print value counts for categorical columns.
    """

    categorical_columns = [
        "education",
        "self_employed",
        "loan_status"
    ]

    print("\n" + "=" * 70)
    print("VALUE COUNTS")
    print("=" * 70)

    for column in categorical_columns:

        print(f"\n{column}")

        print(df[column].value_counts())

# ==========================================================
# Loan Status Distribution
# ==========================================================

def plot_loan_status(df: pd.DataFrame) -> None:
    """Loan approval distribution."""

    plt.figure(figsize=(7, 5))

    sns.countplot(
        data=df,
        x="loan_status",
        palette="Set2"
    )

    plt.title("Loan Status Distribution")
    plt.xlabel("Loan Status")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "loan_status_distribution.png")
    plt.close()


# ==========================================================
# Education vs Loan Status
# ==========================================================

def plot_education(df: pd.DataFrame) -> None:
    """Education vs Loan Status."""

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x="education",
        hue="loan_status"
    )

    plt.title("Education vs Loan Status")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "education_vs_loan.png")
    plt.close()


# ==========================================================
# Self Employed vs Loan Status
# ==========================================================

def plot_self_employed(df: pd.DataFrame) -> None:
    """Self employed analysis."""

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x="self_employed",
        hue="loan_status"
    )

    plt.title("Self Employed vs Loan Status")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "self_employed_vs_loan.png")
    plt.close()


# ==========================================================
# Dependents Distribution
# ==========================================================

def plot_dependents(df: pd.DataFrame) -> None:
    """Dependents distribution."""

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x="no_of_dependents"
    )

    plt.title("Dependents Distribution")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "dependents_distribution.png")
    plt.close()


# ==========================================================
# Numeric Distribution Helper
# ==========================================================

def plot_distribution(
    df: pd.DataFrame,
    column: str,
    filename: str,
    title: str,
) -> None:
    """Histogram with KDE."""

    plt.figure(figsize=(8, 5))

    sns.histplot(
        df[column],
        kde=True,
        bins=30
    )

    plt.title(title)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename)
    plt.close()


# ==========================================================
# Correlation Heatmap
# ==========================================================

def plot_heatmap(df: pd.DataFrame) -> None:
    """Correlation heatmap."""

    numeric_df = df.select_dtypes(include=["int64", "float64"])

    plt.figure(figsize=(12, 8))

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Correlation Heatmap")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "heatmap.png")
    plt.close()


# ==========================================================
# Boxplots
# ==========================================================

def plot_boxplots(df: pd.DataFrame) -> None:
    """Generate boxplots for numeric columns."""

    numeric_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    for column in numeric_columns:

        plt.figure(figsize=(8, 4))

        sns.boxplot(
            x=df[column]
        )

        plt.title(column)

        plt.tight_layout()

        plt.savefig(
            PLOTS_DIR / f"{column}_boxplot.png"
        )

        plt.close()


# ==========================================================
# Pair Plot
# ==========================================================

def plot_pairplot(df: pd.DataFrame) -> None:
    """Pairplot of important features."""

    columns = [
        "income_annum",
        "loan_amount",
        "cibil_score",
        "loan_status",
    ]

    sns.pairplot(
        df[columns],
        hue="loan_status"
    )

    plt.savefig(PLOTS_DIR / "pairplot.png")

    plt.close()


# ==========================================================
# Generate All Distribution Plots
# ==========================================================

def plot_all_distributions(df: pd.DataFrame) -> None:
    """Generate histograms."""

    plot_distribution(
        df,
        "income_annum",
        "income_distribution.png",
        "Income Distribution"
    )

    plot_distribution(
        df,
        "loan_amount",
        "loan_amount_distribution.png",
        "Loan Amount Distribution"
    )

    plot_distribution(
        df,
        "loan_term",
        "loan_term_distribution.png",
        "Loan Term Distribution"
    )

    plot_distribution(
        df,
        "cibil_score",
        "cibil_distribution.png",
        "CIBIL Score Distribution"
    )

    plot_distribution(
        df,
        "residential_assets_value",
        "residential_assets_distribution.png",
        "Residential Assets"
    )

    plot_distribution(
        df,
        "commercial_assets_value",
        "commercial_assets_distribution.png",
        "Commercial Assets"
    )

    plot_distribution(
        df,
        "luxury_assets_value",
        "luxury_assets_distribution.png",
        "Luxury Assets"
    )

    plot_distribution(
        df,
        "bank_asset_value",
        "bank_assets_distribution.png",
        "Bank Assets"
    )

# ==========================================================
# Main Function
# ==========================================================

def main() -> None:
    """Run the complete EDA pipeline."""

    logger.info("=" * 60)
    logger.info("Starting Exploratory Data Analysis...")
    logger.info("=" * 60)

    # Load dataset
    df = load_dataset(DATASET_PATH)

    # Dataset overview
    dataset_overview(df)

    # Data types
    data_types(df)

    # Missing values
    missing_values(df)

    # Duplicate values
    duplicate_values(df)

    # Statistical summary
    statistical_summary(df)

    # Unique values
    unique_values(df)

    # Value counts
    value_counts(df)

    logger.info("Generating visualizations...")

    # Count plots
    plot_loan_status(df)
    plot_education(df)
    plot_self_employed(df)
    plot_dependents(df)

    # Distribution plots
    plot_all_distributions(df)

    # Correlation Heatmap
    plot_heatmap(df)

    # Boxplots
    plot_boxplots(df)

    # Pairplot
    plot_pairplot(df)

    logger.info("=" * 60)
    logger.info("EDA Completed Successfully.")
    logger.info("Plots saved in: %s", PLOTS_DIR)
    logger.info("=" * 60)


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":
    main()