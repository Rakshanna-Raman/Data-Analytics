"""app.py — Streamlit entry point for the Skills vs Placement Analytics app."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from analytics import (
    CGPA_BAND_ORDER,
    COL_BRANCH,
    COL_CGPA,
    COL_CODING_LEVEL,
    COL_COLLEGE_TIER,
    COL_DEGREE,
    COL_GENDER,
    COL_INTERNSHIPS,
    COL_JOB_ROLE,
    COL_PACKAGE_LPA,
    COL_PLACEMENT_STATUS,
    COL_PROJECTS,
    COL_SKILLS_COUNT,
    CODING_LEVEL_ORDER,
    NOT_PLACED_COLOR,
    PLACED_COLOR,
    PROJECTS_BUCKET_ORDER,
    ROLE_PALETTE,
    SKILLS_BUCKET_ORDER,
    TIER_PALETTE,
    band_cgpa,
    branch_placement,
    bucket_projects,
    bucket_skills,
    compute_kpis,
    compute_top_patterns,
    encode_for_correlation,
    find_similar,
    internship_project_heatmap,
    load_data as _load_data,
    package_by,
    placement_by_group,
    placement_rate_by,
    profile_comparison_chart,
    role_stats,
    tier_cgpa_heatmap,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Skills vs Placement Analytics",
    layout="wide",
    page_icon="🎓",
)

# ---------------------------------------------------------------------------
# Cached data loader
# ---------------------------------------------------------------------------

@st.cache_data
def load_data():
    return _load_data("Skills_vs_Placement_Reality.csv")


df = load_data()

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
SECTIONS = [
    "Overview",
    "Skills & Coding",
    "Internships & Projects",
    "CGPA & College Tier",
    "Job Role & Salary",
    "Multi-Factor Insights",
    "Profile Lookup",
]

SECTION_DESCRIPTIONS = {
    "Overview":                "KPIs, placement rate, and breakdowns by tier, degree, and gender.",
    "Skills & Coding":         "How skills count and coding level relate to placement and salary.",
    "Internships & Projects":  "Effect of internship count and project volume on outcomes.",
    "CGPA & College Tier":     "CGPA bands and college tier — alone and in combination.",
    "Job Role & Salary":       "Role distribution, salary ranges, and branch-wise placement.",
    "Multi-Factor Insights":   "Correlation heatmap, cross-tabulations, and key observations.",
    "Profile Lookup":          "Enter a student profile and compare it to the dataset.",
}

with st.sidebar:
    st.title("🎓 Navigation")
    section = st.selectbox("Go to section", SECTIONS)
    st.caption(SECTION_DESCRIPTIONS.get(section, ""))
    st.markdown("---")
    st.caption("Dataset: 300 synthetic student records")

# ---------------------------------------------------------------------------
# Footer helper
# ---------------------------------------------------------------------------
FOOTER = "⚠️ Data is synthetic. All conclusions apply only to this dataset."


def render_footer():
    st.markdown("---")
    st.caption(FOOTER)


# ---------------------------------------------------------------------------
# Section rendering
# ---------------------------------------------------------------------------

if section == "Overview":
    st.title("📊 Overview")

    kpis = compute_kpis(df)

    # ------------------------------------------------------------------
    # KPI metric cards
    # ------------------------------------------------------------------
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Students",      kpis["total"])
    c2.metric("Placed",              kpis["placed_count"])
    c3.metric("Placement Rate",      f"{kpis['placement_rate']} %")
    c4.metric("Avg Package (LPA)",   f"₹ {kpis['avg_package']}")
    c5.metric("Median Package (LPA)", f"₹ {kpis['median_package']}")

    st.markdown("---")

    # ------------------------------------------------------------------
    # Row 1: Placed vs Not-Placed donut  |  Placement Rate by College Tier
    # ------------------------------------------------------------------
    row1_left, row1_right = st.columns(2)

    with row1_left:
        st.subheader("Placed vs Not Placed")
        not_placed = kpis["total"] - kpis["placed_count"]
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(
            [kpis["placed_count"], not_placed],
            labels=["Placed", "Not Placed"],
            colors=[PLACED_COLOR, NOT_PLACED_COLOR],
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops={"width": 0.55},   # donut effect
        )
        ax.set_title("Placed vs Not Placed", fontsize=13, fontweight="bold")
        st.pyplot(fig)
        plt.close(fig)

    with row1_right:
        st.subheader("Placement Rate by College Tier")
        tier_df = placement_by_group(df, COL_COLLEGE_TIER)
        colors = [TIER_PALETTE.get(t, "#888888") for t in tier_df[COL_COLLEGE_TIER]]
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.barh(tier_df[COL_COLLEGE_TIER], tier_df["placement_rate"], color=colors)
        ax.set_xlabel("Placement Rate (%)")
        ax.set_ylabel("College Tier")
        ax.set_title("Placement Rate by College Tier", fontsize=13, fontweight="bold")
        ax.set_xlim(0, 100)
        for i, v in enumerate(tier_df["placement_rate"]):
            ax.text(v + 1, i, f"{v}%", va="center", fontsize=10)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Row 2: Placement Rate by Degree  |  Placement Rate by Gender
    # ------------------------------------------------------------------
    row2_left, row2_right = st.columns(2)

    with row2_left:
        st.subheader("Placement Rate by Degree")
        deg_df = placement_by_group(df, COL_DEGREE)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(deg_df[COL_DEGREE], deg_df["placement_rate"], color=PLACED_COLOR)
        ax.set_xlabel("Degree")
        ax.set_ylabel("Placement Rate (%)")
        ax.set_title("Placement Rate by Degree", fontsize=13, fontweight="bold")
        ax.set_ylim(0, 100)
        for i, v in enumerate(deg_df["placement_rate"]):
            ax.text(i, v + 1, f"{v}%", ha="center", fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with row2_right:
        st.subheader("Placement Rate by Gender")
        gen_df = placement_by_group(df, COL_GENDER)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(gen_df[COL_GENDER], gen_df["placement_rate"], color=NOT_PLACED_COLOR)
        ax.set_xlabel("Gender")
        ax.set_ylabel("Placement Rate (%)")
        ax.set_title("Placement Rate by Gender", fontsize=13, fontweight="bold")
        ax.set_ylim(0, 100)
        for i, v in enumerate(gen_df["placement_rate"]):
            ax.text(i, v + 1, f"{v}%", ha="center", fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")
    st.info(
        "This dataset contains 300 synthetic student records. "
        "All patterns and conclusions apply only to this dataset "
        "and should not be generalised."
    )

    render_footer()

elif section == "Skills & Coding":
    st.title("🧠 Skills & Coding Proficiency")

    # Pre-compute bucketed DataFrame and Categorical coding_level
    df_bucketed = bucket_skills(df)
    df_bucketed["skills_bucket"] = pd.Categorical(
        df_bucketed["skills_bucket"], categories=SKILLS_BUCKET_ORDER, ordered=True
    )
    df_bucketed[COL_CODING_LEVEL] = pd.Categorical(
        df_bucketed[COL_CODING_LEVEL], categories=CODING_LEVEL_ORDER, ordered=True
    )

    # ------------------------------------------------------------------
    # Subsection 1 — Skills Count
    # ------------------------------------------------------------------
    st.subheader("Skills Count Analysis")

    # Row A: box plot (skills_count by placement_status) | bar (placement rate by bucket)
    rowA_left, rowA_right = st.columns(2)

    with rowA_left:
        st.markdown("**Skills Count Distribution by Placement Status**")
        placed_vals = df_bucketed.loc[
            df_bucketed[COL_PLACEMENT_STATUS] == "Placed", COL_SKILLS_COUNT
        ]
        not_placed_vals = df_bucketed.loc[
            df_bucketed[COL_PLACEMENT_STATUS] == "Not Placed", COL_SKILLS_COUNT
        ]
        fig, ax = plt.subplots(figsize=(5, 4))
        bp = ax.boxplot(
            [placed_vals, not_placed_vals],
            tick_labels=["Placed", "Not Placed"],
            patch_artist=True,
        )
        bp["boxes"][0].set_facecolor(PLACED_COLOR)
        bp["boxes"][1].set_facecolor(NOT_PLACED_COLOR)
        for element in ("medians",):
            for line in bp[element]:
                line.set_color("white")
                line.set_linewidth(2)
        ax.set_ylabel("Skills Count")
        ax.set_title("Skills Count by Placement Status", fontsize=12, fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with rowA_right:
        st.markdown("**Placement Rate (%) by Skills Bucket**")
        rate_bucket = placement_rate_by(df_bucketed, "skills_bucket")
        # Ensure Low → Mid → High order
        rate_bucket["skills_bucket"] = pd.Categorical(
            rate_bucket["skills_bucket"], categories=SKILLS_BUCKET_ORDER, ordered=True
        )
        rate_bucket = rate_bucket.sort_values("skills_bucket")
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(
            rate_bucket["skills_bucket"].astype(str),
            rate_bucket["placement_rate"],
            color=PLACED_COLOR,
        )
        ax.set_xlabel("Skills Bucket")
        ax.set_ylabel("Placement Rate (%)")
        ax.set_ylim(0, 105)
        ax.set_title("Placement Rate by Skills Bucket", fontsize=12, fontweight="bold")
        for bar, val in zip(bars, rate_bucket["placement_rate"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{val}%",
                ha="center", va="bottom", fontsize=10,
            )
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Row B: median package by skills_bucket (placed only)
    st.markdown("**Median Package (LPA) by Skills Bucket** *(placed students only)*")
    pkg_bucket = package_by(df_bucketed, "skills_bucket")
    pkg_bucket["skills_bucket"] = pd.Categorical(
        pkg_bucket["skills_bucket"], categories=SKILLS_BUCKET_ORDER, ordered=True
    )
    pkg_bucket = pkg_bucket.sort_values("skills_bucket")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(
        pkg_bucket["skills_bucket"].astype(str),
        pkg_bucket["median_package"],
        color=PLACED_COLOR,
    )
    ax.set_xlabel("Skills Bucket")
    ax.set_ylabel("Median Package (LPA)")
    ax.set_title("Median Package by Skills Bucket (Placed Only)", fontsize=12, fontweight="bold")
    for bar, val in zip(bars, pkg_bucket["median_package"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"₹{val}",
            ha="center", va="bottom", fontsize=10,
        )
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    # ------------------------------------------------------------------
    # Subsection 2 — Coding Level
    # ------------------------------------------------------------------
    st.subheader("Coding Level Analysis")

    rowB_left, rowB_right = st.columns(2)

    with rowB_left:
        st.markdown("**Placement Rate (%) by Coding Level**")
        rate_coding = placement_rate_by(df_bucketed, COL_CODING_LEVEL)
        rate_coding[COL_CODING_LEVEL] = pd.Categorical(
            rate_coding[COL_CODING_LEVEL], categories=CODING_LEVEL_ORDER, ordered=True
        )
        rate_coding = rate_coding.sort_values(COL_CODING_LEVEL)
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(
            rate_coding[COL_CODING_LEVEL].astype(str),
            rate_coding["placement_rate"],
            color=PLACED_COLOR,
        )
        ax.set_xlabel("Coding Level")
        ax.set_ylabel("Placement Rate (%)")
        ax.set_ylim(0, 105)
        ax.set_title("Placement Rate by Coding Level", fontsize=12, fontweight="bold")
        for bar, val in zip(bars, rate_coding["placement_rate"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{val}%",
                ha="center", va="bottom", fontsize=10,
            )
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with rowB_right:
        st.markdown("**Package (LPA) Distribution by Coding Level** *(placed only)*")
        placed_df = df_bucketed[df_bucketed[COL_PLACEMENT_STATUS] == "Placed"].copy()
        placed_df[COL_CODING_LEVEL] = pd.Categorical(
            placed_df[COL_CODING_LEVEL], categories=CODING_LEVEL_ORDER, ordered=True
        )
        coding_groups = [
            placed_df.loc[
                placed_df[COL_CODING_LEVEL] == lvl, "package_lpa"
            ].dropna().values
            for lvl in CODING_LEVEL_ORDER
        ]
        fig, ax = plt.subplots(figsize=(5, 4))
        bp = ax.boxplot(
            coding_groups,
            tick_labels=CODING_LEVEL_ORDER,
            patch_artist=True,
        )
        colors_coding = ["#AED9E0", "#5FA8D3", "#1B4F72"]
        for box, c in zip(bp["boxes"], colors_coding):
            box.set_facecolor(c)
        for median_line in bp["medians"]:
            median_line.set_color("white")
            median_line.set_linewidth(2)
        ax.set_xlabel("Coding Level")
        ax.set_ylabel("Package (LPA)")
        ax.set_title("Package Distribution by Coding Level (Placed Only)", fontsize=11, fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")

    # ------------------------------------------------------------------
    # Callout metric — highest placement rate bucket
    # ------------------------------------------------------------------
    rate_bucket_sorted = rate_bucket.sort_values("placement_rate", ascending=False)
    top_bucket = str(rate_bucket_sorted.iloc[0]["skills_bucket"])
    top_rate = rate_bucket_sorted.iloc[0]["placement_rate"]
    st.metric(
        label="🏆 Skills Bucket with Highest Placement Rate",
        value=top_bucket,
        delta=f"{top_rate}% placement rate",
    )

    render_footer()

elif section == "Internships & Projects":
    st.title("🏗️ Internships & Projects")

    df_proj = bucket_projects(df)
    df_proj["projects_bucket"] = pd.Categorical(
        df_proj["projects_bucket"],
        categories=PROJECTS_BUCKET_ORDER,
        ordered=True,
    )

    # ------------------------------------------------------------------
    # Subsection 1 — Internships
    # ------------------------------------------------------------------
    st.subheader("Internships Analysis")

    int_left, int_right = st.columns(2)

    with int_left:
        st.markdown("**Placement Rate (%) by Internships Count**")
        rate_intern = placement_rate_by(df_proj, COL_INTERNSHIPS)
        rate_intern = rate_intern.sort_values(COL_INTERNSHIPS)
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(
            rate_intern[COL_INTERNSHIPS].astype(str),
            rate_intern["placement_rate"],
            color=PLACED_COLOR,
        )
        ax.set_xlabel("Number of Internships")
        ax.set_ylabel("Placement Rate (%)")
        ax.set_ylim(0, 105)
        ax.set_title("Placement Rate by Internships Count", fontsize=12, fontweight="bold")
        for bar, val in zip(bars, rate_intern["placement_rate"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{val}%",
                ha="center", va="bottom", fontsize=10,
            )
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with int_right:
        st.markdown("**Package (LPA) Distribution by Internships Count** *(placed only)*")
        placed_df = df_proj[df_proj[COL_PLACEMENT_STATUS] == "Placed"]
        intern_counts = sorted(placed_df[COL_INTERNSHIPS].unique())
        intern_groups = [
            placed_df.loc[placed_df[COL_INTERNSHIPS] == n, COL_PACKAGE_LPA].dropna().values
            for n in intern_counts
        ]
        fig, ax = plt.subplots(figsize=(5, 4))
        bp = ax.boxplot(
            intern_groups,
            tick_labels=[str(n) for n in intern_counts],
            patch_artist=True,
        )
        colors_intern = ["#AED9E0", "#5FA8D3", "#1B4F72", "#0D2B45"]
        for box, c in zip(bp["boxes"], colors_intern[: len(intern_counts)]):
            box.set_facecolor(c)
        for median_line in bp["medians"]:
            median_line.set_color("white")
            median_line.set_linewidth(2)
        ax.set_xlabel("Number of Internships")
        ax.set_ylabel("Package (LPA)")
        ax.set_title("Package Distribution by Internships Count (Placed Only)", fontsize=11, fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")

    # ------------------------------------------------------------------
    # Subsection 2 — Projects
    # ------------------------------------------------------------------
    st.subheader("Projects Analysis")

    proj_left, proj_right = st.columns(2)

    with proj_left:
        st.markdown("**Placement Rate (%) by Projects Bucket**")
        rate_proj = placement_rate_by(df_proj, "projects_bucket")
        rate_proj["projects_bucket"] = pd.Categorical(
            rate_proj["projects_bucket"],
            categories=PROJECTS_BUCKET_ORDER,
            ordered=True,
        )
        rate_proj = rate_proj.sort_values("projects_bucket")
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(
            rate_proj["projects_bucket"].astype(str),
            rate_proj["placement_rate"],
            color=PLACED_COLOR,
        )
        ax.set_xlabel("Projects Bucket")
        ax.set_ylabel("Placement Rate (%)")
        ax.set_ylim(0, 105)
        ax.set_title("Placement Rate by Projects Bucket", fontsize=12, fontweight="bold")
        for bar, val in zip(bars, rate_proj["placement_rate"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{val}%",
                ha="center", va="bottom", fontsize=10,
            )
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with proj_right:
        st.markdown("**Projects Count vs Package (LPA)** *(placed only, coloured by college tier)*")
        placed_proj = df_proj[df_proj[COL_PLACEMENT_STATUS] == "Placed"]
        fig, ax = plt.subplots(figsize=(5, 4))
        for tier, color in TIER_PALETTE.items():
            subset = placed_proj[placed_proj[COL_COLLEGE_TIER] == tier]
            ax.scatter(
                subset[COL_PROJECTS],
                subset[COL_PACKAGE_LPA],
                label=tier,
                color=color,
                alpha=0.7,
                edgecolors="white",
                linewidths=0.4,
                s=60,
            )
        ax.set_xlabel("Number of Projects")
        ax.set_ylabel("Package (LPA)")
        ax.set_title("Projects vs Package (Placed Only)", fontsize=12, fontweight="bold")
        ax.legend(title="College Tier", fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")

    # ------------------------------------------------------------------
    # Subsection 3 — Combined Heatmap
    # ------------------------------------------------------------------
    st.subheader("Internships × Projects Heatmap")

    pivot = internship_project_heatmap(df_proj)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor="#e5e7eb",
        ax=ax,
        cbar_kws={"label": "Placement Rate (%)"},
    )
    ax.set_title(
        "Placement Rate (%) by Internships × Projects Volume",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Projects Volume")
    ax.set_ylabel("Internships Count")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    # ------------------------------------------------------------------
    # Callout — internship count with highest placement rate
    # ------------------------------------------------------------------
    best_row = rate_intern.loc[rate_intern["placement_rate"].idxmax()]
    best_n = int(best_row[COL_INTERNSHIPS])
    best_rate = best_row["placement_rate"]
    st.success(
        f"🏆 **{best_n} internship(s)** is associated with the highest placement rate "
        f"in this dataset: **{best_rate}%**"
    )

    render_footer()

elif section == "CGPA & College Tier":
    st.title("CGPA & College Tier")

    # Prepare banded DataFrame once
    df_cgpa = band_cgpa(df)
    placed_cgpa = df_cgpa[df_cgpa[COL_PLACEMENT_STATUS] == "Placed"].copy()

    # -----------------------------------------------------------------------
    # Subsection 1 — CGPA
    # -----------------------------------------------------------------------
    st.subheader("CGPA")
    cgpa_left, cgpa_right = st.columns(2)

    with cgpa_left:
        # Box plot: CGPA distribution by placement status
        fig, ax = plt.subplots(figsize=(6, 4))
        placed_vals_cgpa = df_cgpa.loc[
            df_cgpa[COL_PLACEMENT_STATUS] == "Placed", COL_CGPA
        ].dropna()
        not_placed_vals_cgpa = df_cgpa.loc[
            df_cgpa[COL_PLACEMENT_STATUS] == "Not Placed", COL_CGPA
        ].dropna()
        bp = ax.boxplot(
            [placed_vals_cgpa, not_placed_vals_cgpa],
            tick_labels=["Placed", "Not Placed"],
            patch_artist=True,
        )
        bp["boxes"][0].set_facecolor(PLACED_COLOR)
        bp["boxes"][1].set_facecolor(NOT_PLACED_COLOR)
        for median_line in bp["medians"]:
            median_line.set(color="white", linewidth=2)
        ax.set_title("CGPA Distribution by Placement Status")
        ax.set_ylabel("CGPA")
        ax.set_xlabel("Placement Status")
        st.pyplot(fig)
        plt.close(fig)

    with cgpa_right:
        # Bar chart: placement rate % by CGPA band (ordered)
        rate_band = placement_rate_by(df_cgpa, "cgpa_band")
        # Reindex to enforce CGPA_BAND_ORDER
        rate_band = (
            rate_band.set_index("cgpa_band")
            .reindex(CGPA_BAND_ORDER)
            .reset_index()
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(
            rate_band["cgpa_band"],
            rate_band["placement_rate"],
            color=PLACED_COLOR,
        )
        for bar, val in zip(bars, rate_band["placement_rate"]):
            if not __import__("math").isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.8,
                    f"{val:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        ax.set_title("Placement Rate by CGPA Band")
        ax.set_xlabel("CGPA Band")
        ax.set_ylabel("Placement Rate (%)")
        ax.set_ylim(0, 110)
        st.pyplot(fig)
        plt.close(fig)

    # -----------------------------------------------------------------------
    # Subsection 2 — College Tier
    # -----------------------------------------------------------------------
    st.subheader("College Tier")
    tier_left, tier_right = st.columns(2)

    with tier_left:
        # Bar chart: placement rate % by college_tier
        rate_tier = placement_rate_by(df_cgpa, COL_COLLEGE_TIER)
        tier_order = ["Tier-1", "Tier-2", "Tier-3"]
        rate_tier = (
            rate_tier.set_index(COL_COLLEGE_TIER)
            .reindex(tier_order)
            .reset_index()
        )
        tier_colors = [TIER_PALETTE.get(t, "#888888") for t in rate_tier[COL_COLLEGE_TIER]]
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(rate_tier[COL_COLLEGE_TIER], rate_tier["placement_rate"], color=tier_colors)
        for bar, val in zip(bars, rate_tier["placement_rate"]):
            if not __import__("math").isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.8,
                    f"{val:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        ax.set_title("Placement Rate by College Tier")
        ax.set_xlabel("College Tier")
        ax.set_ylabel("Placement Rate (%)")
        ax.set_ylim(0, 110)
        st.pyplot(fig)
        plt.close(fig)

    with tier_right:
        # Box plot: package_lpa by college_tier (placed only)
        fig, ax = plt.subplots(figsize=(6, 4))
        tier_groups = [
            placed_cgpa.loc[placed_cgpa[COL_COLLEGE_TIER] == t, COL_PACKAGE_LPA].dropna()
            for t in tier_order
        ]
        bp = ax.boxplot(tier_groups, tick_labels=tier_order, patch_artist=True)
        for box, tier in zip(bp["boxes"], tier_order):
            box.set_facecolor(TIER_PALETTE.get(tier, "#888888"))
        for median_line in bp["medians"]:
            median_line.set(color="white", linewidth=2)
        ax.set_title("Package (LPA) by College Tier — Placed Students")
        ax.set_xlabel("College Tier")
        ax.set_ylabel("Package (LPA)")
        st.pyplot(fig)
        plt.close(fig)

    # -----------------------------------------------------------------------
    # Subsection 3 — Tier × CGPA Interaction
    # -----------------------------------------------------------------------
    st.subheader("Tier × CGPA Interaction")
    interact_left, interact_right = st.columns(2)

    with interact_left:
        # Scatter plot: cgpa vs package_lpa (placed only), coloured by college_tier
        fig, ax = plt.subplots(figsize=(6, 4))
        for tier, color in TIER_PALETTE.items():
            subset = placed_cgpa[placed_cgpa[COL_COLLEGE_TIER] == tier]
            ax.scatter(
                subset[COL_CGPA],
                subset[COL_PACKAGE_LPA],
                label=tier,
                color=color,
                alpha=0.65,
                edgecolors="white",
                linewidths=0.4,
                s=45,
            )
        ax.set_title("CGPA vs Package (LPA) — Placed Students")
        ax.set_xlabel("CGPA")
        ax.set_ylabel("Package (LPA)")
        ax.legend(title="College Tier")
        st.pyplot(fig)
        plt.close(fig)

    with interact_right:
        # Seaborn heatmap: tier × CGPA band placement rate
        pivot = tier_cgpa_heatmap(df_cgpa)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".1f",
            cmap="YlGnBu",
            linewidths=0.5,
            ax=ax,
            cbar_kws={"label": "Placement Rate (%)"},
        )
        ax.set_title("Placement Rate (%) by College Tier × CGPA Band")
        ax.set_xlabel("CGPA Band")
        ax.set_ylabel("College Tier")
        st.pyplot(fig)
        plt.close(fig)

    # -----------------------------------------------------------------------
    # Observation note — computed from data, not hardcoded
    # -----------------------------------------------------------------------
    pivot_obs = tier_cgpa_heatmap(df_cgpa)

    # Which tier has the highest overall placement rate?
    overall_tier_rates = placement_rate_by(df_cgpa, COL_COLLEGE_TIER)
    best_tier_row = overall_tier_rates.loc[overall_tier_rates["placement_rate"].idxmax()]
    best_tier_name = best_tier_row[COL_COLLEGE_TIER]
    best_tier_rate = best_tier_row["placement_rate"]

    # Does Tier-1 outperform at low CGPA (< 7.0)?
    low_band = "< 7.0"
    tier1_low = (
        pivot_obs.loc["Tier-1", low_band]
        if "Tier-1" in pivot_obs.index and low_band in pivot_obs.columns
        else None
    )
    other_tiers_low = [
        pivot_obs.loc[t, low_band]
        for t in pivot_obs.index
        if t != "Tier-1" and low_band in pivot_obs.columns
        and not __import__("math").isnan(pivot_obs.loc[t, low_band])
    ]
    if tier1_low is not None and not __import__("math").isnan(float(tier1_low)) and other_tiers_low:
        tier1_leads_low = float(tier1_low) >= max(float(v) for v in other_tiers_low)
        tier1_low_str = (
            f"Tier-1 **does** still lead at low CGPA (< 7.0) with a "
            f"**{tier1_low:.1f}%** placement rate in that band."
            if tier1_leads_low
            else f"At low CGPA (< 7.0) Tier-1 (**{tier1_low:.1f}%**) does "
            f"**not** outperform every other tier — high CGPA alone may "
            f"partially compensate for tier disadvantage."
        )
    else:
        tier1_low_str = "Insufficient data in the < 7.0 CGPA band for Tier-1 to draw a conclusion."

    st.info(
        f"🏆 **{best_tier_name}** has the highest overall placement rate in this dataset "
        f"(**{best_tier_rate:.1f}%**).  \n{tier1_low_str}"
    )

    render_footer()

elif section == "Job Role & Salary":
    st.title("Job Role & Salary")

    placed_df = df[df[COL_PACKAGE_LPA] > 0].copy()
    rs = role_stats(df)

    # ------------------------------------------------------------------
    # Subsection 1 — Role Distribution
    # ------------------------------------------------------------------
    st.subheader("Role Distribution")
    role_left, role_right = st.columns(2)

    with role_left:
        # Horizontal bar chart: count of placements by job_role
        role_counts = (
            rs.set_index(COL_JOB_ROLE)["count"]
            .sort_values(ascending=True)  # ascending so largest is at top
        )
        bar_colors = [
            ROLE_PALETTE[i % len(ROLE_PALETTE)]
            for i in range(len(role_counts))
        ]
        fig, ax = plt.subplots(figsize=(7, max(4, len(role_counts) * 0.5)))
        ax.barh(role_counts.index, role_counts.values, color=bar_colors)
        ax.set_xlabel("Number of Placed Students")
        ax.set_title("Placements by Job Role")
        for i, v in enumerate(role_counts.values):
            ax.text(v + 0.2, i, str(v), va="center", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with role_right:
        # role_stats table
        display_rs = rs.rename(columns={
            COL_JOB_ROLE: "Job Role",
            "count": "Placements",
            "mean_package": "Mean LPA",
            "median_package": "Median LPA",
        })
        st.dataframe(
            display_rs,
            column_config={
                "Mean LPA":   st.column_config.NumberColumn(format="%.2f LPA"),
                "Median LPA": st.column_config.NumberColumn(format="%.2f LPA"),
            },
            use_container_width=True,
            hide_index=True,
        )

    # ------------------------------------------------------------------
    # Subsection 2 — Salary Analysis
    # ------------------------------------------------------------------
    st.subheader("Salary Analysis")

    sal_row1_left, sal_row1_right = st.columns(2)

    with sal_row1_left:
        # Box plot: package_lpa by job_role, sorted by median (highest at top)
        role_order_median = (
            rs.sort_values("median_package", ascending=True)[COL_JOB_ROLE].tolist()
        )
        box_groups = [
            placed_df.loc[placed_df[COL_JOB_ROLE] == r, COL_PACKAGE_LPA].values
            for r in role_order_median
        ]
        fig, ax = plt.subplots(figsize=(7, max(4, len(role_order_median) * 0.55)))
        bp = ax.boxplot(
            box_groups,
            orientation="horizontal",
            patch_artist=True,
            tick_labels=role_order_median,
        )
        for i, (box, median_line) in enumerate(
            zip(bp["boxes"], bp["medians"])
        ):
            color = ROLE_PALETTE[i % len(ROLE_PALETTE)]
            box.set_facecolor(color)
            box.set_alpha(0.7)
            median_line.set_color("black")
            median_line.set_linewidth(2)
        ax.set_xlabel("Package (LPA)")
        ax.set_title("Salary Distribution by Job Role")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with sal_row1_right:
        # Salary histogram with mean and median lines
        mean_sal = placed_df[COL_PACKAGE_LPA].mean()
        median_sal = placed_df[COL_PACKAGE_LPA].median()
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(
            placed_df[COL_PACKAGE_LPA],
            bins=20,
            color=PLACED_COLOR,
            edgecolor="white",
            alpha=0.85,
        )
        ax.axvline(mean_sal, color="#E84855", linestyle="--", linewidth=2,
                   label=f"Mean: {mean_sal:.2f} LPA")
        ax.axvline(median_sal, color="#F4A261", linestyle="-.", linewidth=2,
                   label=f"Median: {median_sal:.2f} LPA")
        ax.set_xlabel("Package (LPA)")
        ax.set_ylabel("Number of Students")
        ax.set_title("Salary Distribution (Placed Students)")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Average package by job_role bar chart (full width)
    avg_pkg = (
        rs[[COL_JOB_ROLE, "mean_package"]]
        .sort_values("mean_package", ascending=False)
    )
    avg_colors = [
        ROLE_PALETTE[i % len(ROLE_PALETTE)] for i in range(len(avg_pkg))
    ]
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(avg_pkg[COL_JOB_ROLE], avg_pkg["mean_package"], color=avg_colors)
    ax.set_ylabel("Average Package (LPA)")
    ax.set_title("Average Package by Job Role")
    ax.tick_params(axis="x", rotation=30)
    for bar, val in zip(bars, avg_pkg["mean_package"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=9,
        )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Subsection 3 — Branch Analysis
    # ------------------------------------------------------------------
    st.subheader("Branch Analysis")

    branch_df = branch_placement(df)
    branch_colors = [
        ROLE_PALETTE[i % len(ROLE_PALETTE)] for i in range(len(branch_df))
    ]
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(
        branch_df[COL_BRANCH], branch_df["placement_rate"], color=branch_colors
    )
    ax.set_ylabel("Placement Rate (%)")
    ax.set_title("Placement Rate by Branch")
    ax.set_ylim(0, 110)
    ax.tick_params(axis="x", rotation=30)
    for bar, val in zip(bars, branch_df["placement_rate"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.1f}%",
            ha="center", va="bottom", fontsize=9,
        )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Top-3 roles callout
    # ------------------------------------------------------------------
    top3 = rs.head(3)[COL_JOB_ROLE].tolist()
    top3_str = ", ".join(
        f"**{r}** ({rs.loc[rs[COL_JOB_ROLE] == r, 'median_package'].values[0]:.2f} LPA)"
        for r in top3
    )
    st.success(f"🏆 Top-3 job roles by median package in this dataset: {top3_str}")

    render_footer()

elif section == "Multi-Factor Insights":
    st.title("Multi-Factor Insights & Correlations")

    # ------------------------------------------------------------------
    # Subsection 1 — Correlation Heatmap
    # ------------------------------------------------------------------
    st.subheader("Feature Correlation Matrix")
    st.caption(
        "placement_status encoded as 1 = Placed, 0 = Not Placed. "
        "college_tier encoded as Tier-1=1, Tier-2=2, Tier-3=3. "
        "coding_level encoded as Basic=1, Intermediate=2, Advanced=3."
    )

    num_df = encode_for_correlation(df)
    corr_matrix = num_df.corr()

    fig_hm, ax_hm = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.4,
        ax=ax_hm,
    )
    ax_hm.set_title(
        "Feature Correlation Matrix\n(placement_status encoded as 1=Placed, 0=Not Placed)",
        fontsize=13,
        pad=12,
    )
    plt.tight_layout()
    st.pyplot(fig_hm)
    plt.close(fig_hm)

    # ------------------------------------------------------------------
    # Subsection 2 — Cross-Tabulation Charts
    # ------------------------------------------------------------------
    st.subheader("Cross-Tabulation: Placement Rate by Combined Factors")

    cross_left, cross_right = st.columns(2)

    # Chart A — placement rate by coding_level × college_tier
    with cross_left:
        ct_df = df.copy()
        ct_grp = (
            ct_df.groupby([COL_CODING_LEVEL, COL_COLLEGE_TIER], observed=True)[COL_PLACEMENT_STATUS]
            .apply(lambda s: round((s == "Placed").sum() / len(s) * 100, 1))
            .reset_index()
        )
        ct_grp.columns = [COL_CODING_LEVEL, COL_COLLEGE_TIER, "placement_rate"]

        fig_a, ax_a = plt.subplots(figsize=(6, 5))
        tier_list = ["Tier-1", "Tier-2", "Tier-3"]
        tier_colors_ct = ["#3A86FF", "#8338EC", "#FF006E"]
        coding_positions = {lvl: i for i, lvl in enumerate(CODING_LEVEL_ORDER)}
        n_tiers = len(tier_list)
        bar_width = 0.25
        offsets = [bar_width * (i - (n_tiers - 1) / 2) for i in range(n_tiers)]

        for tier, color, offset in zip(tier_list, tier_colors_ct, offsets):
            subset = ct_grp[ct_grp[COL_COLLEGE_TIER] == tier]
            xs = [coding_positions.get(lvl, idx) + offset
                  for idx, lvl in enumerate(subset[COL_CODING_LEVEL])]
            ax_a.bar(xs, subset["placement_rate"], width=bar_width,
                     label=tier, color=color, edgecolor="white")

        ax_a.set_xticks(range(len(CODING_LEVEL_ORDER)))
        ax_a.set_xticklabels(CODING_LEVEL_ORDER)
        ax_a.set_xlabel("Coding Level")
        ax_a.set_ylabel("Placement Rate (%)")
        ax_a.set_title("Placement Rate:\nCoding Level × College Tier")
        ax_a.legend(title="College Tier")
        ax_a.set_ylim(0, 105)
        plt.tight_layout()
        st.pyplot(fig_a)
        plt.close(fig_a)

    # Chart B — placement rate by internships × skills_bucket
    with cross_right:
        df_sk_ct = bucket_skills(df)
        is_grp = (
            df_sk_ct.groupby([COL_INTERNSHIPS, "skills_bucket"], observed=True)[COL_PLACEMENT_STATUS]
            .apply(lambda s: round((s == "Placed").sum() / len(s) * 100, 1))
            .reset_index()
        )
        is_grp.columns = [COL_INTERNSHIPS, "skills_bucket", "placement_rate"]

        fig_b, ax_b = plt.subplots(figsize=(6, 5))
        bucket_list = SKILLS_BUCKET_ORDER
        bucket_colors = ["#F4A261", "#2A9D8F", "#264653"]
        intern_vals = sorted(df_sk_ct[COL_INTERNSHIPS].unique())
        n_buckets = len(bucket_list)
        bar_width_b = 0.22
        offsets_b = [bar_width_b * (i - (n_buckets - 1) / 2) for i in range(n_buckets)]

        for bucket, color, offset in zip(bucket_list, bucket_colors, offsets_b):
            subset = is_grp[is_grp["skills_bucket"] == bucket]
            xs = [list(intern_vals).index(v) + offset for v in subset[COL_INTERNSHIPS]]
            ax_b.bar(xs, subset["placement_rate"], width=bar_width_b,
                     label=str(bucket), color=color, edgecolor="white")

        ax_b.set_xticks(range(len(intern_vals)))
        ax_b.set_xticklabels([str(v) for v in intern_vals])
        ax_b.set_xlabel("Number of Internships")
        ax_b.set_ylabel("Placement Rate (%)")
        ax_b.set_title("Placement Rate:\nInternships × Skills Bucket")
        ax_b.legend(title="Skills Bucket")
        ax_b.set_ylim(0, 105)
        plt.tight_layout()
        st.pyplot(fig_b)
        plt.close(fig_b)

    # ------------------------------------------------------------------
    # Subsection 3 — Key Takeaways
    # ------------------------------------------------------------------
    st.subheader("📌 Key Observations (computed from this dataset)")

    for observation in compute_top_patterns(df):
        st.success(observation)

    render_footer()

elif section == "Profile Lookup":
    st.title("🎓 Student Profile Lookup")
    st.markdown(
        "Enter a student profile below and see how it compares to similar students "
        "in the dataset. Results are filtered from the 300 synthetic records."
    )

    unique_degrees = sorted(df[COL_DEGREE].unique().tolist())
    unique_branches = sorted(df[COL_BRANCH].unique().tolist())

    with st.form("profile_form"):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            p_skills   = st.slider("Skills Count",   min_value=2,   max_value=10,  value=5,   step=1)
            p_intern   = st.selectbox("Internships", options=[0, 1, 2, 3], index=1)
            p_projects = st.slider("Projects",       min_value=1,   max_value=6,   value=3,   step=1)
        with col_b:
            p_cgpa     = st.slider("CGPA",           min_value=6.0, max_value=9.5, value=7.5, step=0.1)
            p_coding   = st.selectbox("Coding Level", options=CODING_LEVEL_ORDER, index=1)
            p_tier     = st.selectbox("College Tier", options=["Tier-1", "Tier-2", "Tier-3"], index=1)
        with col_c:
            p_degree   = st.selectbox("Degree",  options=unique_degrees)
            p_branch   = st.selectbox("Branch",  options=unique_branches)
            p_gender   = st.selectbox("Gender",  options=["M", "F"])
        submitted = st.form_submit_button("🔍 Analyse Profile")

    if submitted:
        profile_dict = {
            COL_SKILLS_COUNT:      p_skills,
            COL_INTERNSHIPS:       p_intern,
            COL_PROJECTS:          p_projects,
            COL_CGPA:              p_cgpa,
            COL_CODING_LEVEL:      p_coding,
            COL_COLLEGE_TIER:      p_tier,
            COL_DEGREE:            p_degree,
            COL_BRANCH:            p_branch,
            COL_GENDER:            p_gender,
        }

        matches, relaxed = find_similar(df, profile_dict)

        if len(matches) == 0:
            st.error("No similar students found in the dataset even after relaxing filters.")
        else:
            if relaxed:
                st.warning(
                    "No exact matches found within the numeric tolerances. "
                    "Showing students with matching background (degree, branch, tier, "
                    "coding level, gender only)."
                )

            # ----------------------------------------------------------------
            # Metrics row
            # ----------------------------------------------------------------
            placed_matches = matches[matches[COL_PACKAGE_LPA] > 0]
            match_placement_rate = len(placed_matches) / len(matches) * 100

            if len(placed_matches) > 0:
                common_role = placed_matches[COL_JOB_ROLE].mode().iloc[0]
                median_pkg  = placed_matches[COL_PACKAGE_LPA].median()
            else:
                common_role = "—"
                median_pkg  = 0.0

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Matching Students",   len(matches))
            m2.metric("Placement Rate",       f"{match_placement_rate:.1f}%")
            m3.metric("Most Common Role",     common_role)
            m4.metric("Median Package (LPA)", f"{median_pkg:.1f}" if median_pkg > 0 else "—")

            # ----------------------------------------------------------------
            # Comparison chart
            # ----------------------------------------------------------------
            st.subheader("Profile vs Placed Student Averages")
            fig_cmp = profile_comparison_chart(df, profile_dict)
            st.pyplot(fig_cmp)
            plt.close(fig_cmp)

            # ----------------------------------------------------------------
            # Matched records table
            # ----------------------------------------------------------------
            st.subheader(f"Matched Records ({len(matches)} students)")
            display_cols = [
                COL_DEGREE, COL_BRANCH, COL_COLLEGE_TIER, COL_CODING_LEVEL,
                COL_SKILLS_COUNT, COL_INTERNSHIPS, COL_PROJECTS, COL_CGPA,
                COL_PLACEMENT_STATUS, COL_JOB_ROLE, COL_PACKAGE_LPA,
            ]
            st.dataframe(
                matches[display_cols].reset_index(drop=True),
                use_container_width=True,
            )

    render_footer()
