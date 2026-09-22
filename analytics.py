"""analytics.py — data loading, cleaning, and shared constants for the
Skills vs Placement analytics app."""

from pandas.core.series import Series
from typing import Any
from pandas.core.frame import DataFrame
from numpy import dtype, ndarray
from numpy._typing._shape import _AnyShape
import pandas as pd
import numpy as np
from typing import cast
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
    salary_values = np.asarray(salary_df, dtype=float)

    avg_package    = round(float(np.mean(salary_values)), 2) if len(salary_values) else 0.0
    median_package = round(float(np.median(salary_values)), 2) if len(salary_values) else 0.0
    min_package    = round(float(np.min(salary_values)), 2) if len(salary_values) else 0.0
    max_package    = round(float(np.max(salary_values)), 2) if len(salary_values) else 0.0

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


# ---------------------------------------------------------------------------
# Internships & Projects helpers
# ---------------------------------------------------------------------------

# Ordered bucket labels (Low → Mid → High)
PROJECTS_BUCKET_ORDER = ["Low (1–2)", "Mid (3–4)", "High (5–6)"]


def bucket_projects(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with a new ``projects_bucket`` column.

    Bands:
        1–2 → "Low (1–2)"
        3–4 → "Mid (3–4)"
        5–6 → "High (5–6)"
    """
    out = df.copy()
    bins = [0, 2, 4, 6]
    labels = PROJECTS_BUCKET_ORDER
    out["projects_bucket"] = pd.cut(
        out[COL_PROJECTS],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )
    return out


def internship_project_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """Return a pivot table of placement rate (%) by internships × projects_bucket.

    Rows    = internships count  (0, 1, 2, 3)
    Columns = projects_bucket    (Low (1–2) → Mid (3–4) → High (5–6))
    Values  = placement rate %   (0–100)

    The input *df* must already contain a ``projects_bucket`` column (produced
    by :func:`bucket_projects`).  Columns are ordered Low → Mid → High.
    """
    df_pb = df if "projects_bucket" in df.columns else bucket_projects(df)

    placed = (df_pb[COL_PLACEMENT_STATUS] == "Placed").astype(int)
    pivot = (
        df_pb.assign(placed=placed)
        .groupby([COL_INTERNSHIPS, "projects_bucket"], observed=True)["placed"]
        .apply(lambda s: round(s.mean() * 100, 1))
        .unstack("projects_bucket")
    )
    # Ensure column order Low → Mid → High (only keep buckets present)
    ordered_cols: list[str] = [b for b in PROJECTS_BUCKET_ORDER if b in pivot.columns]
    pivot = cast(pd.DataFrame, pivot[ordered_cols])
    return pivot


# ---------------------------------------------------------------------------
# Job Role & Salary helpers
# ---------------------------------------------------------------------------

def role_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-role placement counts and salary stats (placed students only).

    Filters to rows where ``package_lpa > 0``, groups by ``job_role``, and
    computes count, mean_package, and median_package.

    Returns:
        DataFrame with columns [job_role, count, mean_package, median_package],
        sorted by median_package descending.
    """
    placed = df[df[COL_PACKAGE_LPA] > 0].copy()
    stats = (
        placed.groupby(COL_JOB_ROLE)[COL_PACKAGE_LPA]
        .agg(count="count", mean_package="mean", median_package="median")
        .round(2)
        .reset_index()
    )
    return stats.sort_values("median_package", ascending=False).reset_index(drop=True)


def branch_placement(df: pd.DataFrame) -> pd.DataFrame:
    """Return placement rate (%) per branch, sorted descending.

    Returns:
        DataFrame with columns [branch, placement_rate].
    """
    grouped = (
        df.groupby(COL_BRANCH)[COL_PLACEMENT_STATUS]
        .apply(lambda s: round((s == "Placed").sum() / len(s) * 100, 1))
        .reset_index()
    )
    grouped.columns = [COL_BRANCH, "placement_rate"]
    return grouped.sort_values("placement_rate", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# CGPA & College Tier helpers
# ---------------------------------------------------------------------------

# Ordered CGPA band labels
CGPA_BAND_ORDER = ["< 7.0", "7.0–7.9", "8.0–8.9", "≥ 9.0"]


def band_cgpa(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with a new ``cgpa_band`` column.

    Bands:
        < 7.0  → "< 7.0"
        7.0–7.9 → "7.0–7.9"
        8.0–8.9 → "8.0–8.9"
        ≥ 9.0  → "≥ 9.0"
    """
    out = df.copy()
    bins = [0.0, 6.9999, 7.9999, 8.9999, 10.0]
    labels = CGPA_BAND_ORDER
    out["cgpa_band"] = pd.cut(
        out[COL_CGPA],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )
    return out


def tier_cgpa_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """Return a pivot table of placement rate (%) by college_tier × cgpa_band.

    Rows    = college_tier  (Tier-1, Tier-2, Tier-3)
    Columns = cgpa_band     (ordered per CGPA_BAND_ORDER)
    Values  = placement rate % (0–100)

    If *df* does not yet have a ``cgpa_band`` column it is added automatically.
    """
    df_b = df if "cgpa_band" in df.columns else band_cgpa(df)

    placed = (df_b[COL_PLACEMENT_STATUS] == "Placed").astype(int)
    pivot = (
        df_b.assign(placed=placed)
        .groupby([COL_COLLEGE_TIER, "cgpa_band"], observed=True)["placed"]
        .apply(lambda s: round(s.mean() * 100, 1))
        .unstack("cgpa_band")
    )
    # Enforce column order; keep only bands that are present in the data
    ordered_cols: list[str] = [b for b in CGPA_BAND_ORDER if b in pivot.columns]
    pivot = cast(pd.DataFrame, pivot[ordered_cols])
    return pivot


# ---------------------------------------------------------------------------
# Multi-Factor Insights helpers
# ---------------------------------------------------------------------------

def encode_for_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode categorical columns and return a numeric-only DataFrame.

    Encodings applied:
        placement_status : Placed=1, Not Placed=0
        gender           : ordinal integer via factorize
        degree           : ordinal integer via factorize
        branch           : ordinal integer via factorize
        college_tier     : Tier-1=1, Tier-2=2, Tier-3=3
        coding_level     : Basic=1, Intermediate=2, Advanced=3

    All other numeric columns are kept as-is.  Non-numeric / non-encoded
    columns (student_id, job_role) are dropped.

    Returns:
        A numeric-only DataFrame suitable for :meth:`~pd.DataFrame.corr`.
    """
    out = df.copy()

    # Binary target
    out[COL_PLACEMENT_STATUS] = (out[COL_PLACEMENT_STATUS] == "Placed").astype(int)

    # Ordered categoricals
    out[COL_COLLEGE_TIER] = out[COL_COLLEGE_TIER].map(
    lambda x: 1 if x == "Tier-1" else 2 if x == "Tier-2" else 3
)
    out[COL_CODING_LEVEL] = out[COL_CODING_LEVEL].map(
    lambda x: 1 if x == "Basic" else 2 if x == "Intermediate" else 3
)

    # Nominal categoricals — simple factorize (arbitrary integer codes)
    out[COL_GENDER], _ = pd.factorize(out[COL_GENDER])
    out[COL_DEGREE], _ = pd.factorize(out[COL_DEGREE])
    out[COL_BRANCH], _ = pd.factorize(out[COL_BRANCH])

    # Drop columns that remain non-numeric or carry no signal for correlation
    drop_cols = [COL_STUDENT_ID, COL_JOB_ROLE]
    out = out.drop(columns=[c for c in drop_cols if c in out.columns])

    return out.select_dtypes(include="number")


def compute_top_patterns(df: pd.DataFrame) -> list:
    """Return a list of human-readable observation strings derived from the data.

    For each key dimension the function finds the group with the highest and
    lowest placement rate and formats the result as a sentence.  All values
    are computed from *df* — nothing is hardcoded.

    Dimensions covered:
        skills_bucket, coding_level, internships, cgpa_band,
        college_tier, degree

    Returns:
        A list of ~6–8 strings, one per dimension comparison.
    """
    observations = []

    def _hi_lo(rate_df: pd.DataFrame, col: str) -> tuple:
        """Return (hi_label, hi_rate, lo_label, lo_rate) for a rate DataFrame."""
        hi = rate_df.loc[rate_df["placement_rate"].idxmax()]
        lo = rate_df.loc[rate_df["placement_rate"].idxmin()]
        return str(hi[col]), float(hi["placement_rate"]), str(lo[col]), float(lo["placement_rate"])

    # 1. Skills bucket
    df_sk = bucket_skills(df)
    rate_sk = placement_rate_by(df_sk, "skills_bucket")
    hi_lbl, hi_r, lo_lbl, lo_r = _hi_lo(rate_sk, "skills_bucket")
    observations.append(
        f"Students in the {hi_lbl} skills bucket had a placement rate of "
        f"{hi_r:.1f}%, vs {lo_r:.1f}% for the {lo_lbl} bucket."
    )

    # 2. Coding level
    rate_cl = placement_rate_by(df, COL_CODING_LEVEL)
    hi_lbl, hi_r, lo_lbl, lo_r = _hi_lo(rate_cl, COL_CODING_LEVEL)
    observations.append(
        f"{hi_lbl} coders were placed at {hi_r:.1f}%, "
        f"compared to {lo_r:.1f}% for {lo_lbl} coders."
    )

    # 3. Internships
    rate_int = placement_rate_by(df, COL_INTERNSHIPS)
    hi_lbl, hi_r, lo_lbl, lo_r = _hi_lo(rate_int, COL_INTERNSHIPS)
    observations.append(
        f"Students with {hi_lbl} internship(s) had the highest placement rate "
        f"({hi_r:.1f}%), while those with {lo_lbl} internship(s) had the lowest "
        f"({lo_r:.1f}%)."
    )

    # 4. CGPA band
    df_cg = band_cgpa(df)
    rate_cg = placement_rate_by(df_cg, "cgpa_band")
    hi_lbl, hi_r, lo_lbl, lo_r = _hi_lo(rate_cg, "cgpa_band")
    observations.append(
        f"The {hi_lbl} CGPA band achieved the highest placement rate "
        f"({hi_r:.1f}%), vs {lo_r:.1f}% for the {lo_lbl} band."
    )

    # 5. College tier
    rate_tier = placement_rate_by(df, COL_COLLEGE_TIER)
    hi_lbl, hi_r, lo_lbl, lo_r = _hi_lo(rate_tier, COL_COLLEGE_TIER)
    observations.append(
        f"{hi_lbl} colleges had the best placement rate ({hi_r:.1f}%), "
        f"ahead of {lo_lbl} ({lo_r:.1f}%)."
    )

    # 6. Degree
    rate_deg = placement_rate_by(df, COL_DEGREE)
    hi_lbl, hi_r, lo_lbl, lo_r = _hi_lo(rate_deg, COL_DEGREE)
    observations.append(
        f"{hi_lbl} graduates were placed at {hi_r:.1f}%, "
        f"vs {lo_r:.1f}% for {lo_lbl} graduates."
    )

    # 7. Combined best: skills_bucket × coding_level
    df_sk2 = bucket_skills(df)
    grp = (
        df_sk2.groupby(["skills_bucket", COL_CODING_LEVEL], observed=True)[COL_PLACEMENT_STATUS]
        .apply(lambda s: round((s == "Placed").sum() / len(s) * 100, 1))
        .reset_index()
    )
    grp.columns = ["skills_bucket", COL_CODING_LEVEL, "placement_rate"]
    best_combo = grp.loc[grp["placement_rate"].idxmax()]
    observations.append(
        f"The strongest combination observed was {best_combo['skills_bucket']} skills "
        f"+ {best_combo[COL_CODING_LEVEL]} coding, with a placement rate of "
        f"{best_combo['placement_rate']:.1f}%."
    )

    return observations
