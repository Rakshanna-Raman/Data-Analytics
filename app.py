"""app.py — Streamlit entry point for the Skills vs Placement Analytics app."""

import matplotlib.pyplot as plt
import streamlit as st

from analytics import (
    COL_CODING_LEVEL,
    COL_COLLEGE_TIER,
    COL_DEGREE,
    COL_GENDER,
    COL_PLACEMENT_STATUS,
    COL_SKILLS_COUNT,
    CODING_LEVEL_ORDER,
    NOT_PLACED_COLOR,
    PLACED_COLOR,
    SKILLS_BUCKET_ORDER,
    TIER_PALETTE,
    bucket_skills,
    compute_kpis,
    load_data as _load_data,
    package_by,
    placement_by_group,
    placement_rate_by,
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

with st.sidebar:
    st.title("🎓 Navigation")
    section = st.selectbox("Go to section", SECTIONS)

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
    import pandas as pd
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
            labels=["Placed", "Not Placed"],
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
            labels=CODING_LEVEL_ORDER,
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
    st.title("Internships & Projects")
    st.info("Coming soon...")
    render_footer()

elif section == "CGPA & College Tier":
    st.title("CGPA & College Tier")
    st.info("Coming soon...")
    render_footer()

elif section == "Job Role & Salary":
    st.title("Job Role & Salary")
    st.info("Coming soon...")
    render_footer()

elif section == "Multi-Factor Insights":
    st.title("Multi-Factor Insights")
    st.info("Coming soon...")
    render_footer()

elif section == "Profile Lookup":
    st.title("Profile Lookup")
    st.info("Coming soon...")
    render_footer()
