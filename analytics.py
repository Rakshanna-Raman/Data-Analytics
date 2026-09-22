"""analytics.py — data loading, cleaning, and shared constants for the
Skills vs Placement analytics app."""

import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Seaborn theme — applied once at module import time
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------------------
# Column-name constants
# ---------------------------------------------------------------------------
COL_STUDENT_ID      = "student_id"
COL_AGE             = "age"
COL_GENDER          = "gender"
COL_DEGREE          = "degree"
COL_BRANCH          = "branch"
COL_COLLEGE_TIER    = "college_tier"
COL_SKILLS_COUNT    = "skills_count"
COL_INTERNSHIPS     = "internships"
COL_PROJECTS        = "projects"
COL_CODING_LEVEL    = "coding_level"
COL_CGPA            = "cgpa"
COL_PLACEMENT_STATUS = "placement_status"
COL_JOB_ROLE        = "job_role"
COL_PACKAGE_LPA     = "package_lpa"

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
PLACED_COLOR     = "#2E86AB"
NOT_PLACED_COLOR = "#E84855"

TIER_PALETTE = {
    "Tier-1": "#3A86FF",
    "Tier-2": "#8338EC",
    "Tier-3": "#FF006E",
}

ROLE_PALETTE = [
    "#264653", "#2A9D8F", "#E9C46A", "#F4A261",
    "#E76F51", "#457B9D", "#A8DADC", "#1D3557",
    "#6D6875", "#B5838D",
]

# ---------------------------------------------------------------------------
# Data loading & cleaning
# ---------------------------------------------------------------------------

def load_data(path: str = "Skills_vs_Placement_Reality.csv") -> pd.DataFrame:
    """Read the CSV, cast column types, and return a clean DataFrame.

    Args:
        path: Path to the CSV file. Defaults to the project-root CSV.

    Returns:
        Cleaned pandas DataFrame with correct dtypes.
    """
    df = pd.read_csv(path)

    # Integer columns
    df[COL_INTERNSHIPS]  = df[COL_INTERNSHIPS].astype(int)
    df[COL_PROJECTS]     = df[COL_PROJECTS].astype(int)
    df[COL_SKILLS_COUNT] = df[COL_SKILLS_COUNT].astype(int)

    # Float columns
    df[COL_PACKAGE_LPA] = df[COL_PACKAGE_LPA].astype(float)
    df[COL_CGPA]        = df[COL_CGPA].astype(float)

    return df


# ---------------------------------------------------------------------------
# KPI helpers
# ---------------------------------------------------------------------------

def compute_kpis(df: pd.DataFrame) -> dict:
    """Return a dict of top-level KPIs for the Overview dashboard.

    Keys: total, placed_count, placement_rate, avg_package,
          median_package, min_package, max_package.
    Salary stats are computed only over rows where package_lpa > 0.
    """
    total = len(df)
    placed_count = int((df[COL_PLACEMENT_STATUS] == "Placed").sum())
    placement_rate = round(placed_count / total * 100, 1) if total else 0.0

    salary_df = df[df[COL_PACKAGE_LPA] > 0][COL_PACKAGE_LPA]
    avg_package    = round(float(salary_df.mean()),   2) if len(salary_df) else 0.0
    median_package = round(float(salary_df.median()), 2) if len(salary_df) else 0.0
    min_package    = round(float(salary_df.min()),    2) if len(salary_df) else 0.0
    max_package    = round(float(salary_df.max()),    2) if len(salary_df) else 0.0

    return {
        "total":           total,
        "placed_count":    placed_count,
        "placement_rate":  placement_rate,
        "avg_package":     avg_package,
        "median_package":  median_package,
        "min_package":     min_package,
        "max_package":     max_package,
    }


def placement_by_group(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Return placement rate (%) per category of *col*, sorted descending.

    Args:
        df:  The cleaned DataFrame.
        col: Column to group by (e.g. COL_COLLEGE_TIER, COL_DEGREE).

    Returns:
        DataFrame with columns [col, 'placement_rate'], sorted by
        placement_rate descending.
    """
    grouped = (
        df.groupby(col)[COL_PLACEMENT_STATUS]
        .apply(lambda s: round((s == "Placed").sum() / len(s) * 100, 1))
        .reset_index()
    )
    grouped.columns = [col, "placement_rate"]
    return grouped.sort_values("placement_rate", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Skills & Coding helpers
# ---------------------------------------------------------------------------

# Ordered bucket labels (Low → Mid → High)
SKILLS_BUCKET_ORDER = ["Low (2–4)", "Mid (5–7)", "High (8–10)"]

# Ordered coding-level labels
CODING_LEVEL_ORDER = ["Basic", "Intermediate", "Advanced"]


def bucket_skills(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with a new ``skills_bucket`` column.

    Bands:
        2–4  → "Low (2–4)"
        5–7  → "Mid (5–7)"
        8–10 → "High (8–10)"
    """
    out = df.copy()
    bins = [1, 4, 7, 10]
    labels = SKILLS_BUCKET_ORDER
    out["skills_bucket"] = pd.cut(
        out[COL_SKILLS_COUNT],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )
    return out


def placement_rate_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Generic helper: placement rate (%) grouped by *col*.

    Args:
        df:  The cleaned DataFrame (may contain any column set).
        col: Column to group by.

    Returns:
        DataFrame with columns [col, 'placement_rate'] in the original
        groupby order (not sorted), preserving category order when *col*
        is a Categorical.
    """
    grouped = (
        df.groupby(col, observed=True)[COL_PLACEMENT_STATUS]
        .apply(lambda s: round((s == "Placed").sum() / len(s) * 100, 1))
        .reset_index()
    )
    grouped.columns = [col, "placement_rate"]
    return grouped


def package_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Return mean and median package grouped by *col* (placed students only).

    Filters to rows where ``package_lpa > 0`` before aggregating.

    Args:
        df:  The cleaned DataFrame.
        col: Column to group by.

    Returns:
        DataFrame with columns [col, 'mean_package', 'median_package'].
    """
    placed = df[df[COL_PACKAGE_LPA] > 0]
    grouped = (
        placed.groupby(col, observed=True)[COL_PACKAGE_LPA]
        .agg(mean_package="mean", median_package="median")
        .round(2)
        .reset_index()
    )
    return grouped
