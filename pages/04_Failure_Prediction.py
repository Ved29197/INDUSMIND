import pandas as pd
import plotly.express as px
import streamlit as st

from src.classification.artifact import load_failure_model, load_metadata
from src.classification.scoring import (
    engineering_checks, risk_category, typical_values, what_if_contributions,
)
from src.classification.splits import stratified_split
from src.data_processing import schema
from src.data_processing.loader import SAMPLE_PATH, load_ai4i
from src.evaluation.thresholds import theoretical_threshold

st.set_page_config(page_title="Failure Prediction | INDUSMIND", page_icon="🏭", layout="wide")


@st.cache_resource
def get_model():
    return load_failure_model()


@st.cache_data
def get_metadata():
    return load_metadata()


@st.cache_data
def get_demo():
    df, _ = load_ai4i(SAMPLE_PATH)
    _, X_test, _, y_test = stratified_split(df)
    return df, X_test, y_test


st.title("Failure Prediction")
st.warning(
    "**Estimated risk, not a prediction that a machine will fail.** A 10% risk means that "
    "about 1 in 10 similar machines failed in the data. The model is trained on public "
    "**synthetic** data, so results demonstrate the method and say nothing about a real plant."
)

try:
    model = get_model()
    meta = get_metadata()
except FileNotFoundError as err:
    st.error(str(err))
    st.stop()

df_all, X_test, y_test = get_demo()
base_rate = meta["base_failure_rate"]

# ---------------------------------------------------------------- sidebar: costs
st.sidebar.header("Decision setting")
ratio = st.sidebar.slider(
    "Cost of a missed failure / cost of an unneeded inspection (illustrative)", 2, 100, 10
)
threshold = theoretical_threshold(ratio, 1)
st.sidebar.markdown(
    f"Inspect when estimated risk is at least **{threshold:.1%}**. "
    "Threshold = inspection cost / (inspection cost + failure cost)."
)

recorded = meta.get("recorded_test_results", {})
with st.expander("About this model"):
    st.markdown(
        f"- **Model:** {meta['model']} with {meta['calibration']} probability calibration, "
        "using sensor readings plus three physical features (temperature gap, mechanical power, "
        "wear x torque).\n"
        f"- **Trained on:** {meta['training_rows']:,} machines ({meta['training_failures']} failures; "
        f"average failure rate {base_rate:.1%}).\n"
        + (f"- **Held-out test result:** PR-AUC {recorded['test_pr_auc']} "
           f"(95% interval {recorded['test_pr_auc_ci_low']} to {recorded['test_pr_auc_ci_high']}).\n"
           if recorded else "")
        + "- **Limits:** " + " ".join(meta["limitations"])
    )

tab_single, tab_fleet = st.tabs(["Single machine", "Fleet scoring (held-out machines)"])

# ------------------------------------------------------------------ single machine
with tab_single:
    st.subheader("Enter a machine's current readings")
    c1, c2, c3 = st.columns(3)
    machine_type = c1.selectbox("Product type", ["L", "M", "H"])
    air = c1.number_input("Air temperature [K]", 250.0, 350.0, 300.0, 0.1)
    process = c2.number_input("Process temperature [K]", 250.0, 400.0, 310.0, 0.1)
    rpm = c2.number_input("Rotational speed [rpm]", 0, 10000, 1500, 10)
    torque = c3.number_input("Torque [Nm]", 0.0, 200.0, 40.0, 0.5)
    wear = c3.number_input("Tool wear [min]", 0, 500, 100, 5)

    row = pd.DataFrame(
        [{
            "Type": machine_type,
            "Air temperature [K]": air,
            "Process temperature [K]": process,
            "Rotational speed [rpm]": rpm,
            "Torque [Nm]": torque,
            "Tool wear [min]": wear,
        }]
    )[schema.FEATURE_COLUMNS]

    risk = float(model.predict_calibrated(row)[0])
    category = risk_category(risk, threshold, base_rate)

    m1, m2, m3 = st.columns(3)
    m1.metric("Estimated failure risk", f"{risk:.1%}")
    m2.metric("Compared with the average machine", f"{risk / base_rate:.1f}x")
    m3.metric("Risk category", category)
    if category == "Inspect":
        st.error(f"Risk is at or above the {threshold:.1%} cost-based threshold: schedule an inspection.")
    elif category == "Elevated":
        st.warning("Risk is above the average failure rate but below the inspection threshold: monitor.")
    else:
        st.success("Risk is below the average failure rate.")

    st.subheader("What is driving this risk?")
    contributions = what_if_contributions(model, row, typical_values(df_all[schema.FEATURE_COLUMNS]))
    chart = contributions.rename("risk change").reset_index()
    chart.columns = ["Input", "Risk change"]
    chart = chart.reindex(chart["Risk change"].abs().sort_values().index)
    st.plotly_chart(
        px.bar(chart, x="Risk change", y="Input", orientation="h",
               title="Risk change versus a typical machine, one input at a time")
    )
    st.caption(
        "Positive bars show inputs that raise this machine's risk compared with a typical value. "
        "This is a simple what-if view, not a full explanation; inputs interact, so bars need not "
        "add up to the total. Detailed SHAP explanations arrive in a later phase."
    )

    st.subheader("Engineering limit checks")
    checks = engineering_checks(row.iloc[0].to_dict())
    st.dataframe(checks, hide_index=True)
    st.caption(
        "Reference limits come from the AI4I dataset documentation and illustrate engineering "
        "overrides. For a real plant, use the equipment's own design limits."
    )
    if (checks["Status"] == "ALERT").any() and category == "Low":
        st.info(
            "An engineering limit is breached although the model's estimate is low. The model can only "
            "recognise failure modes it has seen, so treat the limit breach as an override."
        )

# ------------------------------------------------------------------ fleet scoring
with tab_fleet:
    st.subheader("Scoring 2,000 machines the model never saw during training")
    scores = model.predict_calibrated(X_test)
    categories = [risk_category(p, threshold, base_rate) for p in scores]
    view = pd.DataFrame(
        {
            "Product ID": df_all.loc[X_test.index, "Product ID"].values,
            "Type": X_test["Type"].values,
            "Estimated risk": scores,
            "Category": categories,
            "Recorded failure (synthetic)": y_test.values,
        }
    ).sort_values("Estimated risk", ascending=False)

    flagged = view["Category"] == "Inspect"
    f1, f2, f3 = st.columns(3)
    f1.metric("Machines flagged Inspect", int(flagged.sum()))
    f2.metric("...of which recorded a failure", int(view.loc[flagged, "Recorded failure (synthetic)"].sum()))
    f3.metric("Failures not flagged", int(view.loc[~flagged, "Recorded failure (synthetic)"].sum()))

    counts = view["Category"].value_counts().reindex(["Low", "Elevated", "Inspect"], fill_value=0)
    st.plotly_chart(
        px.bar(counts.rename("machines").reset_index(), x="Category", y="machines",
               title="Machines per risk category")
    )
    st.markdown("**Highest-risk machines**")
    st.dataframe(
        view.head(15), hide_index=True,
        column_config={"Estimated risk": st.column_config.NumberColumn(format="%.3f")},
    )
    st.download_button(
        "Download all scores (CSV)", view.to_csv(index=False), "indusmind_scores.csv", "text/csv"
    )
    st.caption(
        "The 'recorded failure' column is the synthetic dataset's own label, shown only so you can see "
        "how the estimates compare. Many machines share identical probabilities because of the "
        "calibration method, so equal-risk machines are listed in arbitrary order."
    )