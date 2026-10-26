# Campus Facility Reservation and Monitoring System (Streamlit)

Streamlit version of the **Data Mining Laboratory Examination** notebook. It covers
Parts I to V: data understanding, cleaning and feature engineering, exploratory
analysis, Decision Tree modelling, functional requirements and the dashboard.

## Folder contents

| File | Purpose |
| --- | --- |
| `app.py` | The Streamlit application |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

## Data

The notebook read `facility_reservations.csv` from Google Colab (`/content/`). That file was
not included with the notebook, so the app works in two ways:

1. **Upload your CSV** using the sidebar uploader. This is the option to use for real results.
2. **Synthetic demo data** is used if nothing is uploaded. It has the same columns but is
   generated, so its numbers will not match your real results.

Expected columns: `Reservation_ID`, `Facility_Type`, `User_Category`, `Purpose`,
`Participants`, `Duration_Hours`, `Submission_Date`, `Reservation_Date`, `Lead_Time_Days`,
`Day_Of_Week`, `Facility_ID`, `Usage_Status`, `Actual_Attendance`, `Cancellation_Days_Before`.

## Sections (sidebar navigation)

- **Part I - Data Understanding:** first 10 records, shape, data types, summary statistics,
  data dictionary, predictors, target and the data acquisition note.
- **Part II - Cleaning & Features:** missing values, duplicates, category standardisation,
  numeric and date validation, `Lead_Time_Category`, `Reservation_Outcome`,
  `Attendance_Ratio`, and a download button for the cleaned CSV.
- **Part III - EDA:** reservations by facility type, cancellation/no-show percentage by user
  category, and lead-time category versus outcome (Figures 1 to 4).
- **Part IV - Decision Tree:** model training, confusion matrix, accuracy, precision, recall,
  F1, top predictors, tree plot and the data leakage check (Figure 5).
- **Part V - Requirements & Dashboard:** functional requirements built from the computed
  findings, the 2x2 dashboard (Figure 7) and the short written answers.

The sidebar also lets you change the maximum tree depth, test-set size and random seed.
The notebook defaults are depth 5, 20% test set and seed 42.

## Differences from the notebook

- Findings in the requirements table and the dashboard are computed from the loaded data
  instead of being typed in (the notebook hard-coded values such as 119 classroom bookings).
- The dashboard's "predicted risk" panel shows Low, Medium and High risk counts from the
  model's predicted probabilities on the test set. The notebook only had a text placeholder.
- Written answer (c) was blank in the notebook, so a suggested answer is included. Review
  and edit it before submitting.
- The UML use case diagram (Figure 6) is an embedded image in the notebook and is not
  part of this app.

## Notes

- `Usage_Status`, `Actual_Attendance` and `Cancellation_Days_Before` are excluded from the
  predictors because they are only known after the outcome (data leakage).
- Predicted risk is advisory and should support administrator review, not reject a
  reservation automatically.
