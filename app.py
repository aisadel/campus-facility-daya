"""
Campus Facility Reservation and Monitoring System
Data Mining Laboratory Examination - Streamlit version

Converted from the notebook: Parts I-V (data understanding, cleaning,
EDA, Decision Tree modelling, functional requirements and dashboard).

Run with:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

st.set_page_config(
    page_title="Facility Reservation Data Mining",
    page_icon="🏫",
    layout="wide",
)

# ----------------------------------------------------------------------
# Constants (taken directly from the notebook)
# ----------------------------------------------------------------------
CATEGORICAL_COLUMNS = [
    "User_Category",
    "Facility_Type",
    "Purpose",
    "Day_Of_Week",
    "Usage_Status",
]
NUMERIC_COLUMNS = [
    "Lead_Time_Days",
    "Participants",
    "Duration_Hours",
    "Actual_Attendance",
    "Cancellation_Days_Before",
]
CANDIDATE_PREDICTORS = [
    "User_Category",
    "Facility_Type",
    "Facility_ID",
    "Purpose",
    "Lead_Time_Days",
    "Day_Of_Week",
    "Participants",
    "Duration_Hours",
]
FORBIDDEN_PREDICTORS = ["Usage_Status", "Actual_Attendance", "Cancellation_Days_Before"]
LEAD_ORDER = ["Last Minute", "Short Lead", "Medium Lead", "Early Booking"]

DATA_DICTIONARY = pd.DataFrame(
    [
        ("Reservation_ID", "Categorical / Identifier", "Unique transaction identifier; not a model predictor"),
        ("Facility_Type", "Categorical", "Predictor candidate: type of facility reserved"),
        ("User_Category", "Categorical", "Predictor candidate: category of user"),
        ("Purpose", "Categorical", "Predictor candidate: purpose of reservation"),
        ("Participants", "Numerical", "Predictor candidate: planned number of participants"),
        ("Duration_Hours", "Numerical", "Predictor candidate: planned reservation duration"),
        ("Submission_Date", "Date", "Date the reservation was submitted"),
        ("Reservation_Date", "Date", "Date of the planned reservation"),
        ("Lead_Time_Days", "Numerical", "Predictor candidate: days between submission and reservation"),
        ("Day_Of_Week", "Categorical", "Predictor candidate: scheduled day of the week"),
        ("Facility_ID", "Categorical / Identifier", "Identifies the reserved facility"),
        ("Usage_Status", "Categorical", "Source field for deriving Reservation_Outcome"),
        ("Actual_Attendance", "Numerical", "Observed attendance; post-outcome information"),
        ("Cancellation_Days_Before", "Numerical", "Cancellation timing; post-outcome information"),
    ],
    columns=["Variable", "Type", "Role"],
)


# ----------------------------------------------------------------------
# Demo data (synthetic) - only used when no CSV is uploaded
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def make_demo_data(n: int = 360, seed: int = 42) -> pd.DataFrame:
    """Synthetic data with the same schema as facility_reservations.csv.

    It is NOT the real examination dataset. It deliberately contains a few
    messy values (mixed case, whitespace, duplicates, invalid numbers) so the
    cleaning step has something to demonstrate.
    """
    rng = np.random.default_rng(seed)
    facility_types = ["Classroom", "Laboratory", "Meeting Room", "Sports Facility", "Auditorium"]
    prefix = {"Classroom": "CR", "Laboratory": "LB", "Meeting Room": "MR", "Sports Facility": "SF", "Auditorium": "AU"}
    users = ["Student", "Faculty", "Administrative Staff", "Student Organization"]
    purposes = ["Class", "Meeting", "Event", "Training", "Practice"]

    ftype = rng.choice(facility_types, n, p=[0.33, 0.23, 0.17, 0.17, 0.10])
    user = rng.choice(users, n, p=[0.30, 0.30, 0.20, 0.20])
    purpose = rng.choice(purposes, n)
    participants = rng.integers(5, 120, n)
    duration = rng.choice([1, 2, 3, 4, 6, 8], n)
    submission = pd.Timestamp("2025-01-06") + pd.to_timedelta(rng.integers(0, 150, n), unit="D")
    lead = rng.integers(0, 30, n)
    reservation = submission + pd.to_timedelta(lead, unit="D")
    fac_id = [f"{prefix[t]}-{rng.integers(1, 6):02d}" for t in ftype]

    # Cancellation chance rises with group size and for some user types
    p_cancel = 0.18 + 0.002 * participants + np.where(user == "Student Organization", 0.08, 0.0)
    p_cancel = np.clip(p_cancel, 0.05, 0.6)
    roll = rng.random(n)
    status = np.where(roll < p_cancel * 0.6, "Cancelled", np.where(roll < p_cancel, "No-Show", "Used"))

    attendance = np.where(
        status == "Used",
        np.minimum(participants + rng.integers(-5, 6, n), participants + 3).clip(min=1),
        0,
    )
    cancel_days = np.where(status == "Cancelled", rng.integers(0, 10, n), np.nan)

    df = pd.DataFrame(
        {
            "Reservation_ID": [f"R{1000 + i}" for i in range(n)],
            "Facility_Type": ftype,
            "User_Category": user,
            "Purpose": purpose,
            "Participants": participants,
            "Duration_Hours": duration,
            "Submission_Date": submission.strftime("%Y-%m-%d"),
            "Reservation_Date": reservation.strftime("%Y-%m-%d"),
            "Lead_Time_Days": lead,
            "Day_Of_Week": reservation.day_name(),
            "Facility_ID": fac_id,
            "Usage_Status": status,
            "Actual_Attendance": attendance,
            "Cancellation_Days_Before": cancel_days,
        }
    )

    # Inject a little mess for the cleaning step
    df.loc[rng.choice(n, 6, replace=False), "Usage_Status"] = "cancelled "
    df.loc[rng.choice(n, 5, replace=False), "User_Category"] = "  student"
    df.loc[rng.choice(n, 4, replace=False), "Participants"] = -3
    df.loc[rng.choice(n, 4, replace=False), "Purpose"] = np.nan
    df = pd.concat([df, df.sample(6, random_state=1)], ignore_index=True)
    return df


# ----------------------------------------------------------------------
# Part II - cleaning and feature engineering
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def clean_data(df_original: pd.DataFrame):
    """Return (cleaned_df, report_dict). Mirrors Part II of the notebook."""
    report = {}
    df = df_original.copy()

    report["missing_before"] = df.isnull().sum().to_frame("Missing_Values")
    report["duplicates_found"] = int(df.duplicated().sum())

    # Step 3: standardise text categories
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip().replace("", pd.NA)
            df[col] = df[col].str.title()

    if "Usage_Status" in df.columns:
        status_map = {
            "Used": "Used",
            "Completed": "Used",
            "Cancelled": "Cancelled",
            "Canceled": "Cancelled",
            "No-Show": "No-Show",
            "Noshow": "No-Show",
            "No Show": "No-Show",
        }
        df["Usage_Status"] = df["Usage_Status"].replace(status_map)

    # Step 4: remove exact duplicates
    df = df.drop_duplicates().copy()
    report["rows_after_dedup"] = len(df)

    # Step 5: validate numeric values
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    invalid = {}
    rules = {
        "Lead_Time_Days": ("Negative lead time", lambda s: s < 0),
        "Participants": ("Non-positive participants", lambda s: s <= 0),
        "Duration_Hours": ("Non-positive duration", lambda s: s <= 0),
        "Actual_Attendance": ("Negative actual attendance", lambda s: s < 0),
        "Cancellation_Days_Before": ("Negative cancellation days", lambda s: s < 0),
    }
    for col, (label, rule) in rules.items():
        if col in df.columns:
            bad = rule(df[col])
            invalid[label] = int(bad.sum())
            df.loc[bad, col] = np.nan
    report["invalid_values"] = pd.DataFrame.from_dict(
        invalid, orient="index", columns=["Invalid_Values_Found"]
    )

    # Step 6: validate dates and lead time
    for col in ["Submission_Date", "Reservation_Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    report["date_order_issues"] = 0
    report["lead_time_mismatch"] = 0
    if {"Submission_Date", "Reservation_Date"}.issubset(df.columns):
        calc = (df["Reservation_Date"] - df["Submission_Date"]).dt.days
        neg = calc < 0
        report["date_order_issues"] = int(neg.sum())
        calc = calc.mask(neg)
        if "Lead_Time_Days" in df.columns:
            mismatch = (
                df["Lead_Time_Days"].notna()
                & calc.notna()
                & (df["Lead_Time_Days"] != calc)
            )
            report["lead_time_mismatch"] = int(mismatch.sum())
        df["Lead_Time_Days"] = calc

    # Step 7: Lead_Time_Category
    if "Lead_Time_Days" in df.columns:
        lt = df["Lead_Time_Days"]
        cat = np.select(
            [lt.between(0, 2), lt.between(3, 7), lt.between(8, 14), lt > 14],
            LEAD_ORDER,
            default="Unknown",
        )
        df["Lead_Time_Category"] = pd.Series(cat, index=df.index, dtype="object")
        df.loc[lt.isna(), "Lead_Time_Category"] = pd.NA

    # Step 8: Reservation_Outcome
    if "Usage_Status" in df.columns:
        outcome_map = {"Used": "Completed", "Cancelled": "Not Completed", "No-Show": "Not Completed"}
        df["Reservation_Outcome"] = df["Usage_Status"].map(outcome_map)
        report["unmapped_status"] = int(df["Reservation_Outcome"].isna().sum())

    # Step 9: Attendance_Ratio
    report["attendance_exceeds"] = 0
    if {"Actual_Attendance", "Participants"}.issubset(df.columns):
        valid = df["Participants"] > 0
        df["Attendance_Ratio"] = np.nan
        df.loc[valid, "Attendance_Ratio"] = (
            df.loc[valid, "Actual_Attendance"] / df.loc[valid, "Participants"]
        )
        report["attendance_exceeds"] = int(
            (
                df["Actual_Attendance"].notna()
                & df["Participants"].notna()
                & (df["Actual_Attendance"] > df["Participants"])
            ).sum()
        )

    report["quality_summary"] = pd.DataFrame(
        {"Missing_Values": df.isna().sum(), "Data_Type": df.dtypes.astype(str)}
    )
    report["remaining_duplicates"] = int(df.duplicated().sum())
    return df, report


# ----------------------------------------------------------------------
# Part IV - Decision Tree
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def train_model(df_clean: pd.DataFrame, max_depth: int, test_size: float, seed: int):
    target = "Reservation_Outcome"
    predictors = [c for c in CANDIDATE_PREDICTORS if c in df_clean.columns]

    if target not in df_clean.columns:
        raise ValueError("Reservation_Outcome is missing. The data needs a Usage_Status column.")
    if len(predictors) < 5:
        raise ValueError("Fewer than five candidate predictors are available in the data.")

    model_df = df_clean[df_clean[target].isin(["Completed", "Not Completed"])].copy()
    X = model_df[predictors]
    y = model_df[target]
    if y.nunique() < 2:
        raise ValueError("The target has fewer than two classes; a classifier cannot be trained.")

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [c for c in predictors if c not in numeric_features]

    # Cast text columns to plain object so the sklearn imputer behaves the same
    # across pandas versions
    X = X.copy()
    for c in categorical_features:
        X[c] = X[c].astype(object).where(X[c].notna(), np.nan)

    preprocessor = ColumnTransformer(
        [
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    class_counts = y.value_counts()
    can_stratify = (
        y.nunique() >= 2
        and class_counts.min() >= 2
        and len(y) * test_size >= y.nunique()
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y if can_stratify else None
    )

    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                DecisionTreeClassifier(random_state=seed, max_depth=max_depth, class_weight="balanced"),
            ),
        ]
    )
    model.fit(X_train, y_train)

    return {
        "model": model,
        "predictors": predictors,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_counts": y.value_counts().to_frame("Count"),
    }


def original_feature(encoded_name: str, predictors: list) -> str:
    name = encoded_name.split("__", 1)[-1]
    for feature in sorted(predictors, key=len, reverse=True):
        if name == feature or name.startswith(feature + "_"):
            return feature
    return name


def compute_eda(df: pd.DataFrame) -> dict:
    """Numbers used by Part III, the requirements table and the dashboard."""
    out = {}
    out["facility_counts"] = df["Facility_Type"].dropna().value_counts()

    valid_statuses = ["Used", "Cancelled", "No-Show"]
    sd = df[df["User_Category"].notna() & df["Usage_Status"].isin(valid_statuses)].copy()
    if not sd.empty:
        sd["flag"] = sd["Usage_Status"].isin(["Cancelled", "No-Show"])
        us = sd.groupby("User_Category").agg(
            Eligible_Reservations=("Usage_Status", "size"),
            Cancelled_No_Show_Count=("flag", "sum"),
        )
        us["Cancellation_No_Show_Percentage"] = (
            us["Cancelled_No_Show_Count"] / us["Eligible_Reservations"] * 100
        )
        out["user_summary"] = us.sort_values("Cancellation_No_Show_Percentage", ascending=False)
    else:
        out["user_summary"] = None

    ld = df[
        df["Lead_Time_Category"].notna()
        & df["Reservation_Outcome"].isin(["Completed", "Not Completed"])
    ]
    if not ld.empty:
        oc = pd.crosstab(ld["Lead_Time_Category"], ld["Reservation_Outcome"])
        oc = oc.reindex(columns=["Completed", "Not Completed"], fill_value=0)
        oc = oc.reindex([c for c in LEAD_ORDER if c in oc.index])
        out["outcome_counts"] = oc
        out["outcome_pct"] = oc.div(oc.sum(axis=1).replace(0, np.nan), axis=0) * 100
    else:
        out["outcome_counts"] = None
        out["outcome_pct"] = None
    return out


def get_predictor_importance(bundle: dict) -> pd.Series:
    model = bundle["model"]
    names = model.named_steps["preprocessor"].get_feature_names_out()
    imps = model.named_steps["classifier"].feature_importances_
    df = pd.DataFrame({"Encoded_Feature": names, "Importance": imps})
    df["Original_Predictor"] = df["Encoded_Feature"].apply(
        lambda n: original_feature(n, bundle["predictors"])
    )
    return df.groupby("Original_Predictor")["Importance"].sum().sort_values(ascending=False)


# ----------------------------------------------------------------------
# Sidebar: data source and settings
# ----------------------------------------------------------------------
st.sidebar.title("🏫 Facility Reservations")
st.sidebar.caption("Data Mining Laboratory Examination")

uploaded = st.sidebar.file_uploader("Upload facility_reservations.csv", type=["csv"])
use_demo = False
if uploaded is not None:
    try:
        df_raw = pd.read_csv(uploaded)
    except Exception as exc:  # noqa: BLE001
        st.sidebar.error(f"Could not read the file: {exc}")
        st.stop()
    st.sidebar.success(f"Loaded {uploaded.name}")
else:
    use_demo = st.sidebar.checkbox("Use synthetic demo data", value=True)
    if use_demo:
        df_raw = make_demo_data()
        st.sidebar.warning("Demo data is synthetic, so results will not match your real dataset.")
    else:
        st.info("Upload `facility_reservations.csv` in the sidebar, or tick the demo-data option, to begin.")
        st.stop()

section = st.sidebar.radio(
    "Section",
    [
        "Part I - Data Understanding",
        "Part II - Cleaning & Features",
        "Part III - EDA",
        "Part IV - Decision Tree",
        "Part V - Requirements & Dashboard",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Model settings")
max_depth = st.sidebar.slider("Max tree depth", 2, 10, 5)
test_pct = st.sidebar.slider("Test set size (%)", 10, 40, 20, step=5)
seed = st.sidebar.number_input("Random seed", value=42, step=1)

st.title("Campus Facility Reservation and Monitoring System")
if use_demo and uploaded is None:
    st.warning("Showing results for **synthetic demo data**. Upload your CSV for real results.")

df_clean, report = clean_data(df_raw)


# ----------------------------------------------------------------------
# Part I
# ----------------------------------------------------------------------
if section.startswith("Part I "):
    st.header("Part I - Dataset Acquisition and Understanding")

    st.subheader("Task 1: Dataset overview")
    st.caption(
        "First 10 records, dimensions, variable names and data types, and summary statistics."
    )
    st.markdown("**First 10 records**")
    st.dataframe(df_raw.head(10))

    c1, c2 = st.columns(2)
    c1.metric("Number of rows", df_raw.shape[0])
    c2.metric("Number of columns", df_raw.shape[1])

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Variable names and data types**")
        st.dataframe(
            df_raw.dtypes.astype(str).rename("Data_Type").to_frame()
        )
    with col_b:
        st.markdown("**Variable names (list)**")
        st.code(str(df_raw.columns.tolist()))

    st.markdown("**Summary statistics**")
    st.dataframe(df_raw.describe(include="all").astype(str))

    st.subheader("Task 2: Variable identification")
    st.markdown("**Data dictionary**")
    st.dataframe(DATA_DICTIONARY, hide_index=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**Five potential predictors**")
        st.markdown("1. Facility_Type\n2. User_Category\n3. Purpose\n4. Participants\n5. Duration_Hours")
    with c2:
        st.markdown("**Target variable**")
        st.markdown("`Reservation_Outcome`, derived from `Usage_Status`")
    with c3:
        st.markdown("**Two categorical variables**")
        st.markdown("1. Facility_Type\n2. User_Category")
    with c4:
        st.markdown("**Two numerical variables**")
        st.markdown("1. Participants\n2. Duration_Hours")

    st.subheader("Task 3: Data acquisition")
    st.markdown(
        """
Reservation data could be obtained from the university's reservation system through an
authenticated API or a database query. An API could return reservation records in JSON
format, while a database query could retrieve the required fields from the reservation table.

**Validation check:** Verify that each reservation has a unique `Reservation_ID` and that
required fields contain valid values before accepting the records for analysis.
"""
    )


# ----------------------------------------------------------------------
# Part II
# ----------------------------------------------------------------------
elif section.startswith("Part II "):
    st.header("Part II - Data Cleaning and Feature Engineering")
    st.caption(
        "Checks missing values and duplicates, standardises categories, validates numbers and "
        "dates, and creates Lead_Time_Category, Reservation_Outcome and Attendance_Ratio."
    )

    st.subheader("Step 1-2: Original data and quality checks")
    c1, c2, c3 = st.columns(3)
    c1.metric("Original rows", df_raw.shape[0])
    c2.metric("Original columns", df_raw.shape[1])
    c3.metric("Exact duplicate rows", report["duplicates_found"])
    left, right = st.columns(2)
    with left:
        st.markdown("**Missing values (before cleaning)**")
        st.dataframe(report["missing_before"])
    with right:
        st.markdown("**Original data types**")
        st.dataframe(df_raw.dtypes.astype(str).rename("Data_Type").to_frame())

    st.subheader("Step 3-4: Standardise categories and remove duplicates")
    st.write(
        f"Exact duplicate rows removed: **{report['duplicates_found']}**. "
        f"Rows remaining: **{report['rows_after_dedup']}**."
    )

    st.subheader("Step 5: Numerical validation")
    st.dataframe(report["invalid_values"])

    st.subheader("Step 6: Date validation")
    c1, c2 = st.columns(2)
    c1.metric("Reservation date before submission date", report["date_order_issues"])
    c2.metric("Supplied vs calculated lead time differ", report["lead_time_mismatch"])

    st.subheader("Step 7-9: Engineered features")
    c1, c2, c3 = st.columns(3)
    c1.metric("Unmapped / missing statuses", report.get("unmapped_status", "n/a"))
    c2.metric("Attendance exceeds participants", report["attendance_exceeds"])
    c3.metric("Remaining duplicates", report["remaining_duplicates"])
    st.caption("Records where attendance exceeds participants should be reviewed rather than silently changed.")

    st.markdown("**Cleaned data (first 20 rows)**")
    st.dataframe(df_clean.head(20))

    st.subheader("Step 10: Final data quality summary")
    st.dataframe(report["quality_summary"].astype(str))
    st.write("Final dataset dimensions:", df_clean.shape)

    st.subheader("Step 11: Download cleaned dataset")
    st.download_button(
        "Download facility_reservations_cleaned.csv",
        df_clean.to_csv(index=False).encode("utf-8"),
        file_name="facility_reservations_cleaned.csv",
        mime="text/csv",
    )


# ----------------------------------------------------------------------
# Part III
# ----------------------------------------------------------------------
elif section.startswith("Part III"):
    st.header("Part III - Exploratory Data Analysis (EDA)")

    required = ["Facility_Type", "User_Category", "Usage_Status", "Lead_Time_Category", "Reservation_Outcome"]
    missing = [c for c in required if c not in df_clean.columns]
    if missing:
        st.error(f"Missing required columns: {missing}. Check that the data has the expected schema.")
        st.stop()

    eda = compute_eda(df_clean)

    st.subheader("Reservations by facility type")
    fc = eda["facility_counts"]
    if fc.empty:
        st.write("No valid facility-type records are available.")
    else:
        st.dataframe(fc.rename("Reservation_Count").to_frame())
        st.success(f"Highest reservation count: **{fc.index[0]}** ({fc.iloc[0]} of {len(df_clean)} records)")
        fig, ax = plt.subplots(figsize=(9, 5))
        fc.plot(kind="bar", ax=ax)
        ax.set_title("Part III: Number of Reservations by Facility Type")
        ax.set_xlabel("Facility Type")
        ax.set_ylabel("Number of Reservations")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        fig.tight_layout()
        st.pyplot(fig)
        st.caption("**Figure 1.** Number of reservations by facility type.")

    st.subheader("Cancellation / no-show percentage by user category")
    us = eda["user_summary"]
    if us is None:
        st.write("No reservations with recognised statuses are available.")
    else:
        st.dataframe(us.round(2))
        st.caption(
            "Denominator: reservations with a recognised status (Used, Cancelled or No-Show) "
            "within each user category."
        )
        top = us.iloc[0]
        st.success(
            f"Highest cancellation/no-show percentage: **{us.index[0]}** at "
            f"{top['Cancellation_No_Show_Percentage']:.2f}% "
            f"({int(top['Cancelled_No_Show_Count'])} of {int(top['Eligible_Reservations'])})"
        )
        fig, ax = plt.subplots(figsize=(9, 5))
        us["Cancellation_No_Show_Percentage"].sort_values().plot(kind="barh", ax=ax)
        ax.set_title("Part III: Cancellation/No-Show Percentage by User Category")
        ax.set_xlabel("Cancellation/No-Show Percentage (%)")
        ax.set_ylabel("User Category")
        fig.tight_layout()
        st.pyplot(fig)
        st.caption("**Figure 2.** Cancellation/no-show percentage by user category.")

    st.subheader("Lead-time category vs reservation outcome")
    oc, op = eda["outcome_counts"], eda["outcome_pct"]
    if oc is None:
        st.write("No valid records are available for the lead-time and outcome analysis.")
    else:
        t1, t2 = st.columns(2)
        with t1:
            st.markdown("**Counts**")
            st.dataframe(oc)
        with t2:
            st.markdown("**Percentages within each lead-time category**")
            st.dataframe(op.round(2))
        st.caption(
            "Denominator: all records with a recognised outcome and a known lead-time category, "
            "calculated separately within each lead-time category."
        )

        g1, g2 = st.columns(2)
        with g1:
            fig, ax = plt.subplots(figsize=(7, 5))
            oc.plot(kind="bar", stacked=True, ax=ax)
            ax.set_title("Reservation Outcomes by Lead-Time Category")
            ax.set_xlabel("Lead-Time Category")
            ax.set_ylabel("Number of Reservations")
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
            ax.legend(title="Reservation Outcome")
            fig.tight_layout()
            st.pyplot(fig)
            st.caption("**Figure 3.** Outcomes by lead-time category (counts).")
        with g2:
            fig, ax = plt.subplots(figsize=(7, 5))
            op.plot(kind="bar", stacked=True, ax=ax)
            ax.set_title("Outcome Percentage by Lead-Time Category")
            ax.set_xlabel("Lead-Time Category")
            ax.set_ylabel("Percentage (%)")
            ax.set_ylim(0, 100)
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
            ax.legend(title="Reservation Outcome")
            fig.tight_layout()
            st.pyplot(fig)
            st.caption("**Figure 4.** Outcomes by lead-time category (percentages).")


# ----------------------------------------------------------------------
# Part IV
# ----------------------------------------------------------------------
elif section.startswith("Part IV"):
    st.header("Part IV - Data Mining and Predictive Modeling (Decision Tree)")
    st.caption(
        "Decision Tree classifier predicting Reservation_Outcome, with an 80/20 split by default "
        "(adjustable in the sidebar), evaluation metrics, top predictors and a leakage check."
    )

    try:
        bundle = train_model(df_clean, int(max_depth), test_pct / 100, int(seed))
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    model = bundle["model"]
    y_test = bundle["y_test"]
    X_test = bundle["X_test"]

    st.subheader("Step 1-3: Target, predictors and feature types")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Selected predictors**")
        st.write(bundle["predictors"])
        st.markdown(f"**Target:** `Reservation_Outcome`")
    with c2:
        st.markdown("**Target class counts**")
        st.dataframe(bundle["y_counts"])
    st.write("Numerical features:", bundle["numeric_features"])
    st.write("Categorical features:", bundle["categorical_features"])

    st.subheader("Step 5: Train-test split")
    c1, c2, c3 = st.columns(3)
    c1.metric("Training records", len(bundle["X_train"]))
    c2.metric("Testing records", len(X_test))
    c3.metric("Test-set classes", y_test.nunique())

    st.subheader("Step 7: Evaluation")
    y_pred = model.predict(X_test)
    labels = model.named_steps["classifier"].classes_
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        st.dataframe(
            pd.DataFrame(
                cm,
                index=[f"Actual {l}" for l in labels],
                columns=[f"Predicted {l}" for l in labels],
            ),
        )
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(ax=ax)
        ax.set_title("Decision Tree Confusion Matrix")
        fig.tight_layout()
        st.pyplot(fig)
        st.caption("**Figure 5.** Actual vs predicted reservation outcomes on the test set.")
    with right:
        m1, m2 = st.columns(2)
        m1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
        m2.metric(
            "Weighted F1",
            f"{f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}",
        )
        m3, m4 = st.columns(2)
        m3.metric(
            "Weighted precision",
            f"{precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}",
        )
        m4.metric(
            "Weighted recall",
            f"{recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}",
        )
        st.markdown("**Classification report**")
        st.dataframe(
            pd.DataFrame(
                classification_report(y_test, y_pred, labels=labels, zero_division=0, output_dict=True)
            ).T.round(3),
        )
        if y_test.nunique() < 2:
            st.warning("The test set contains only one actual class, so metrics may be misleading.")

    st.subheader("Step 8: Top influential predictors")
    pi = get_predictor_importance(bundle)
    left, right = st.columns(2)
    with left:
        st.markdown("**Importance by original predictor**")
        st.dataframe(pi.to_frame("Importance").round(4))
    with right:
        st.markdown("**Top three**")
        st.dataframe(pi.head(3).to_frame("Importance").round(4))

    st.subheader("Decision tree visualisation (top levels)")
    tree_model = model.named_steps["classifier"]
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    fig, ax = plt.subplots(figsize=(22, 10))
    plot_tree(
        tree_model,
        feature_names=feature_names,
        class_names=list(tree_model.classes_),
        filled=True,
        rounded=True,
        max_depth=2,
        fontsize=8,
        ax=ax,
    )
    ax.set_title("Decision Tree: Reservation Outcome Prediction", fontsize=14)
    fig.tight_layout()
    st.pyplot(fig)

    st.subheader("Step 9: Data leakage check")
    included = [c for c in bundle["predictors"] if c in FORBIDDEN_PREDICTORS]
    st.write("Excluded to prevent leakage:", FORBIDDEN_PREDICTORS)
    st.write("Forbidden predictors included:", included)
    if not included:
        st.success("Leakage check passed.")
    else:
        st.error("Leakage check failed.")


# ----------------------------------------------------------------------
# Part V
# ----------------------------------------------------------------------
else:
    st.header("Part V - Requirements, Dashboard and Written Answers")

    eda = compute_eda(df_clean)
    try:
        bundle = train_model(df_clean, int(max_depth), test_pct / 100, int(seed))
        model_ok = True
    except ValueError as exc:
        st.error(str(exc))
        model_ok = False

    # ---- Task 9: functional requirements (evidence is computed from the data)
    st.subheader("Task 9: Functional requirements")
    fc = eda["facility_counts"]
    us = eda["user_summary"]
    rows = []
    if not fc.empty:
        rows.append(
            (
                f"Part III: {fc.index[0]} had the highest reservation count, with "
                f"{fc.iloc[0]} reservations out of {len(df_clean)} records.",
                "FR-DM-01",
                "The system shall generate a report showing reservation counts by facility type "
                "and identify the facility type with the highest reservation count.",
            )
        )
    if us is not None:
        top = us.iloc[0]
        rows.append(
            (
                f"Part III: {us.index[0]} had the highest cancellation/no-show percentage, at "
                f"{top['Cancellation_No_Show_Percentage']:.2f}% "
                f"({int(top['Cancelled_No_Show_Count'])} of {int(top['Eligible_Reservations'])} eligible reservations).",
                "FR-DM-02",
                "The system shall calculate and display the cancellation/no-show count and percentage "
                "for each user category, using reservations with recognised statuses as the denominator.",
            )
        )
    if model_ok:
        pi = get_predictor_importance(bundle)
        top3 = ", ".join(f"{n} ({v:.4f})" for n, v in pi.head(3).items())
        rows.append(
            (
                f"Part IV: The Decision Tree ranked {top3} as its top three predictors by "
                "aggregated feature importance.",
                "FR-DM-03",
                "The system shall display the three highest-ranked predictors and their "
                "feature-importance scores from the trained Decision Tree model.",
            )
        )
    st.dataframe(
        pd.DataFrame(rows, columns=["Finding / Evidence", "Requirement ID", "Functional Requirement"]),
        hide_index=True,
    )

    # ---- Dashboard
    st.subheader("Dashboard prototype")
    st.caption(
        "Observed results come from the data. Predicted risk comes from the model on the test set "
        "and is not a confirmed outcome."
    )

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Campus Facility Reservation Dashboard", fontsize=16)

    if not fc.empty:
        ax[0, 0].bar(fc.index.astype(str), fc.values)
    ax[0, 0].set_title("OBSERVED: Reservations by Facility")
    ax[0, 0].set_ylabel("Reservation Count")
    ax[0, 0].tick_params(axis="x", rotation=25)

    if us is not None:
        pct = us["Cancellation_No_Show_Percentage"]
        ax[0, 1].bar(pct.index.astype(str), pct.values)
    ax[0, 1].set_title("OBSERVED: Cancellation/No-Show Rate")
    ax[0, 1].set_ylabel("Rate (%)")
    ax[0, 1].tick_params(axis="x", rotation=25)

    risk_counts = None
    if model_ok:
        classes = list(bundle["model"].named_steps["classifier"].classes_)
        proba = bundle["model"].predict_proba(bundle["X_test"])[:, classes.index("Not Completed")]
        risk = pd.cut(
            pd.Series(proba),
            bins=[-0.01, 0.33, 0.66, 1.0],
            labels=["Low Risk", "Medium Risk", "High Risk"],
        )
        risk_counts = risk.value_counts().reindex(["Low Risk", "Medium Risk", "High Risk"]).fillna(0)
        ax[1, 0].bar(risk_counts.index.astype(str), risk_counts.values)
        ax[1, 0].set_ylabel("Test reservations")
    ax[1, 0].set_title("PREDICTED: Reservation Risk (not a confirmed outcome)")

    if model_ok:
        yp = bundle["model"].predict(bundle["X_test"])
        yt = bundle["y_test"]
        acc = accuracy_score(yt, yp) * 100
        rec = recall_score(yt, yp, pos_label="Not Completed", zero_division=0) * 100
        metrics = {"Test Accuracy": acc, "Not Completed Recall": rec}
        ax[1, 1].bar(list(metrics.keys()), list(metrics.values()))
        ax[1, 1].set_ylim(0, 100)
    ax[1, 1].set_title("MODEL EVALUATION")
    ax[1, 1].set_ylabel("Percentage (%)")
    ax[1, 1].tick_params(axis="x", rotation=10)

    fig.tight_layout()
    st.pyplot(fig)
    st.caption(
        "**Figure 7.** Campus Facility Reservation Dashboard: observed reservations by facility, "
        "observed cancellation/no-show rates, predicted risk, and model evaluation."
    )

    # ---- Task 12 written answers
    st.subheader("Task 12: Short written answers")
    with st.expander("(a) Why should predicted high risk not automatically reject a reservation?", expanded=True):
        st.write(
            "A high-risk prediction is not always correct and may classify a valid reservation as "
            "risky. Automatically rejecting it could unfairly deny facility access. Therefore, "
            "predictions should support administrator review rather than replace human judgment "
            "and reservation policies."
        )
    with st.expander("(b) How does poor data quality affect the model and the information system?", expanded=True):
        st.write(
            "Missing, duplicate, inconsistent, or invalid data can cause inaccurate predictions and "
            "misleading reports. This can lead to poor administrative decisions. Data cleaning, "
            "validation, and regular monitoring help improve the reliability of the model and the "
            "information system."
        )
    with st.expander("(c) What information should be displayed to explain a high-risk prediction?", expanded=True):
        st.write(
            "The notebook left this answer blank. A suggested answer: show the predicted risk level "
            "and probability, the top factors that pushed the reservation towards high risk "
            "(for example group size, lead time, user category or facility), how similar past "
            "reservations turned out, and a note that the prediction is advisory and subject to "
            "administrator review."
        )
