# Skills vs Placement Outcomes — College Student Career Analytics

A data analytics and decision-support application that explores how skills,
coding proficiency, internships, projects, CGPA, and college tier are associated
with student placement outcomes and salary packages.

> **Data note:** The dataset (`Skills_vs_Placement_Reality.csv`) is synthetic.
> All patterns and conclusions apply strictly to this dataset and should not be
> generalised to real-world populations.

---

## Prerequisites

- Python 3.10 or later
- pip

---

## Installation

```bash
# 1. Clone or download the repository
cd skills-placement-analytics

# 2. (Optional but recommended) Create a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running the App

```bash
streamlit run app.py
```

The app will open automatically in your default browser at `http://localhost:8501`.

---

## Project Structure

```
skills-placement-analytics/
├── Skills_vs_Placement_Reality.csv   # Source dataset (300 synthetic student records)
├── app.py                            # Streamlit UI — navigation, layout, widgets
├── analytics.py                      # Data loading, KPI computation, chart helpers
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## Application Sections

| Section | Description |
|---|---|
| **Overview** | Key KPIs (placement rate, avg/median package), breakdowns by college tier, degree, and gender |
| **Skills & Coding** | Placement rate and salary distributions by skills count bucket and coding level |
| **Internships & Projects** | Effect of internship count and project volume; cross-tabulation heatmap |
| **CGPA & College Tier** | CGPA band analysis, tier comparison, and tier × CGPA interaction heatmap |
| **Job Role & Salary** | Role distribution, salary box plots, branch-wise placement rates |
| **Multi-Factor Insights** | Full correlation heatmap, cross-tabulation bar charts, auto-computed key observations |
| **Profile Lookup** | Interactive tool — enter a student profile and compare to similar students in the dataset |

---

## Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Static charts |
| `seaborn` | Statistical visualisation and heatmaps |
| `plotly` | (available for future interactive charts) |
