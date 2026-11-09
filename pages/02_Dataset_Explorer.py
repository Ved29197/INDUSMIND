import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_processing import schema
from src.data_processing.loader import (
    SAMPLE_PATH,
    load_ai4i,
    read_csv_safely,
    validate_ai4i,
)

st.set_page_config(page_title="Dataset Explorer | INDUSMIND", page_icon="🏭", layout="wide")

st.title("Dataset Explorer")
st.warning(
    "Demonstration on the public, **synthetic** UCI AI4I 2020 dataset. "
    "It is not real plant data and says nothing about real plant performance."
)


@st.cache_data
def load_demo():
    return load_ai4i(SAMPLE_PATH)


# ---------------------------------------------------------------- data source
source = st.sidebar.radio("Data source", ["Demo dataset (AI4I 2020)", "Upload my own CSV"])

if source.startswith("Demo"):
    df, report = load_demo()
    is_demo = True
else:
    is_demo = False
    uploaded = st.sidebar.file_uploader("CSV file (AI4I column format)", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV in the AI4I column format, or switch to the demo dataset.")
        with st.expander("Required columns"):
            st.code("\n".join(schema.REQUIRED_COLUMNS))
        st.stop()
    try:
        df = read_csv_safely(uploaded)
    except ValueError as err:
        st.error(str(err))
        st.stop()
    report = validate_ai4i(df)
    if not report.passed:
        st.error("Validation failed. Fix these issues and upload again:")
        for message in report.errors:
            st.write(f"- {message}")
        st.stop()

# ----------------------------------------------------------------------- KPIs
n_rows = len(df)
n_fail = int(df[schema.TARGET].sum())
fail_rate = n_fail / n_rows

c1, c2, c3, c4 = st.columns(4)
c1.metric("Records", f"{n_rows:,}")
c2.metric("Failures", f"{n_fail:,}")
c3.metric("Failure rate", f"{fail_rate:.2%}")
c4.metric("Accuracy of 'always no failure'", f"{1 - fail_rate:.2%}")
st.info(
    f"A useless model that never predicts a failure would score {1 - fail_rate:.1%} "
    "accuracy while catching none of them. That is why accuracy alone is not used "
    "to judge failure-prediction models in this project."
)

outcome = df[schema.TARGET].map({0: "No failure", 1: "Failure"})

tab_quality, tab_dist, tab_rel, tab_fail, tab_about = st.tabs(
    ["Data quality", "Distributions", "Relationships", "Failure analysis", "About the data"]
)

# --------------------------------------------------------------- data quality
with tab_quality:
    if report.warnings:
        for message in report.warnings:
            st.warning(message)
    else:
        st.success("All validation checks passed with no warnings.")

    left, right = st.columns(2)
    with left:
        st.subheader("Missing values")
        missing = df[schema.REQUIRED_COLUMNS].isna().sum().rename("missing values")
        st.dataframe(missing.to_frame())
    with right:
        st.subheader("Statistical summary")
        st.dataframe(df[schema.NUMERIC_FEATURES].describe().T.round(2))

    if is_demo:
        with st.expander("Findings from the exploratory notebook (demo dataset only)"):
            st.markdown(
                "- **Label noise:** 9 failures have no cause flagged, and 18 rows have a "
                "cause flagged (all `RNF`) but no failure.\n"
                "- **Leakage:** the five cause flags alone reproduce the failure label "
                "with 94.8% precision and 97.3% recall, so they are never used as features.\n"
                "- **Deterministic rules:** every `HDF`, `PWF` and `OSF` failure follows a "
                "threshold rule on temperature gap, mechanical power, or wear x torque; "
                "287 of 339 failures (84.7%) are explained this way.\n"
                "- **Hidden time order:** air temperature is smooth along `UDI` "
                "(lag-1 autocorrelation 0.999), and heat-dissipation failures cluster in "
                "rows 4001 to 5000, so `UDI` is excluded from features.\n"
                "- **Stochastic wear failures:** all 46 tool-wear failures occur at wear of "
                "198 minutes or more, but only 46 of 894 such rows fail this way."
            )

# --------------------------------------------------------------- distributions
with tab_dist:
    feature = st.selectbox("Feature", schema.NUMERIC_FEATURES)
    fig = px.histogram(
        df, x=feature, color=outcome, barmode="overlay",
        histnorm="probability density", opacity=0.6, nbins=50,
        labels={"color": "Outcome"}, title=f"{feature}: failures vs. non-failures",
    )
    st.plotly_chart(fig)
    st.caption(
        "Each group is scaled to the same area, so the rare failures stay visible "
        "next to the many non-failures."
    )

# --------------------------------------------------------------- relationships
with tab_rel:
    corr = df[schema.NUMERIC_FEATURES + [schema.TARGET]].corr().round(2)
    fig = px.imshow(
        corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="Correlation matrix",
    )
    st.plotly_chart(fig)
    st.caption(
        "Correlation only measures straight-line relationships. A feature such as tool "
        "wear can show weak correlation and still matter through thresholds."
    )

    x_col, y_col = st.columns(2)
    x_feature = x_col.selectbox("X axis", schema.NUMERIC_FEATURES, index=2)
    y_feature = y_col.selectbox("Y axis", schema.NUMERIC_FEATURES, index=3)
    plot_df = df.assign(Outcome=outcome).sort_values(schema.TARGET)  # failures drawn on top
    fig = px.scatter(
        plot_df, x=x_feature, y=y_feature, color="Outcome", opacity=0.6,
        render_mode="webgl", title=f"{y_feature} vs. {x_feature}",
    )
    st.plotly_chart(fig)

# --------------------------------------------------------------- failure analysis
with tab_fail:
    left, right = st.columns(2)
    with left:
        by_type = df.groupby("Type")[schema.TARGET].agg(rate="mean", rows="count").reset_index()
        st.plotly_chart(px.bar(by_type, x="Type", y="rate", title="Failure rate by product type"))
    with right:
        causes = df[schema.FAILURE_MODE_COLUMNS].sum().rename("count").reset_index()
        causes.columns = ["Cause flag", "count"]
        st.plotly_chart(px.bar(causes, x="Cause flag", y="count", title="Cause flags raised"))

    wear = df["Tool wear [min]"]
    edges = list(range(0, int(wear.max()) + 51, 50))
    band = pd.cut(wear, edges, include_lowest=True)
    by_wear = df.groupby(band, observed=True)[schema.TARGET].agg(rate="mean", rows="count")
    by_wear.index = by_wear.index.astype(str)
    by_wear.index.name = "band"
    st.plotly_chart(
        px.bar(by_wear.reset_index(), x="band", y="rate", hover_data=["rows"],
               title="Failure rate by tool-wear band (minutes)")
    )

    if is_demo:
        block = (df["UDI"] - 1) // 250
        by_block = df.groupby(block)[schema.TARGET].mean()
        by_block.index = [f"{i * 250 + 1}-{(i + 1) * 250}" for i in by_block.index]
        by_block.index.name = "UDI block"
        st.plotly_chart(
            px.bar(by_block.rename("failure rate").reset_index(), x="UDI block",
                   y="failure rate", title="Failure rate per 250-row block of UDI")
        )
        st.caption(
            "Failures cluster in rows 4001 to 5000: the data has a hidden time order. "
            "This is one reason the identifier is never used as a model feature."
        )

# --------------------------------------------------------------------- about
with tab_about:
    st.markdown(
        "**Source:** UCI Machine Learning Repository, AI4I 2020 Predictive Maintenance "
        "Dataset (dataset 601), S. Matzka. About 10,000 synthetic machine records. "
        "Check the licence on the UCI page before reusing it."
    )
    roles = pd.DataFrame(
        {
            "Column(s)": ["UDI, Product ID", "Type", "Air/process temperature, speed, torque, tool wear",
                          "Machine failure", "TWF, HDF, PWF, OSF, RNF"],
            "Role": ["Identifiers", "Feature (categorical)", "Features (numeric)", "Target",
                     "Failure causes (known only after failure)"],
            "Used to predict failure?": ["No", "Yes", "Yes", "No (this is the label)", "No (leakage)"],
        }
    )
    st.dataframe(roles, hide_index=True)
    st.markdown(
        "**Limitations:** the data is synthetic, partly rule-generated, and has a hidden "
        "time order. Results here demonstrate methods and must not be read as validated "
        "real-world performance."
    )