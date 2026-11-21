"""Streamlit dashboard for vote-effective-sample-size.

Three tabs:
1. Audit -- simulate a labelled probe from a population with planted bias
   and audit it: correlation estimate, agreement decomposition, n_eff.
2. Prediction -- predicted majority-vote accuracy vs sample count for the
   correlation-aware predictor and the naive independence baseline, plus
   the accuracy ceiling and the samples-needed solver.
3. Evaluation -- the stored headline results (results/evaluation.json).

Run: streamlit run app.py
"""

import json
import os

import streamlit as st

from vote_eff.audit import run_audit
from vote_eff.models import Population
from vote_eff.predict import accuracy_ceiling, predicted_accuracy, samples_needed
from vote_eff.sampler import generate_probe

st.set_page_config(page_title="Vote Effective Sample Size", layout="wide")


def tab_audit():
    st.header("Audit a probe")
    st.write(
        "A probe is a small labelled set: a few dozen cases, each answered "
        "several times. The audit estimates the error correlation from the "
        "probe's agreement pattern and prices the vote before you pay for "
        "a full self-consistency run."
    )
    col1, col2, col3, col4 = st.columns(4)
    p = col1.slider("True individual accuracy (simulation)", 0.55, 0.95, 0.70, 0.01)
    rho_latent = col2.slider("Planted bias (latent correlation)", 0.0, 0.95, 0.60, 0.05)
    n_cases = col3.slider("Probe cases", 16, 128, 48, 8)
    n_samples = col4.slider("Probe samples per case", 4, 32, 16, 2)
    population = Population(p=p, rho_latent=rho_latent, seed=7)
    probe = generate_probe(population, int(n_cases), int(n_samples), 8)
    result = run_audit(probe)
    st.write("**Recommendation:** %s" % result.recommendation)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Estimated accuracy p_hat", "%.3f" % result.p_hat)
    m2.metric("Estimated correlation rho_hat", "%.3f" % result.rho_hat)
    m3.metric("Bias-driven share of agreement", "%.1f%%" % (100.0 * result.bias_share))
    m4.metric("Accuracy ceiling (any n)", "%.3f" % result.ceiling)
    rows = []
    for row in result.rows:
        rows.append(
            {
                "samples": row.n,
                "effective votes": round(row.n_eff, 2),
                "predicted accuracy": round(row.pred_accuracy, 3),
                "predicted gain": round(row.pred_gain, 3),
                "naive accuracy": round(row.naive_accuracy, 3),
                "naive gain": round(row.naive_gain, 3),
            }
        )
    st.dataframe(rows, use_container_width=True)
    chart = {}
    chart["effective votes"] = [row["effective votes"] for row in rows]
    chart["nominal samples"] = [float(row["samples"]) for row in rows]
    st.bar_chart(chart)
    st.caption(
        "Variance-estimator cross-check for rho: %.3f. True latent bias in "
        "this simulation: %.2f." % (result.rho_hat_variance, rho_latent)
    )


def tab_prediction():
    st.header("Predict the gain before paying for it")
    col1, col2 = st.columns(2)
    p = col1.slider("Individual accuracy p", 0.51, 0.99, 0.70, 0.01)
    rho = col2.slider("Estimated error correlation rho", 0.0, 0.95, 0.40, 0.01)
    ns = list(range(1, 129))
    correlated = [predicted_accuracy(p, rho, n) for n in ns]
    naive = [predicted_accuracy(p, 0.0, n) for n in ns]
    st.line_chart(
        {
            "correlation-aware (n_eff)": correlated,
            "naive independence": naive,
        }
    )
    ceiling = accuracy_ceiling(p, rho)
    st.write(
        "Accuracy ceiling at rho = %.2f: **%.3f** -- no number of samples "
        "passes it while the correlation holds." % (rho, ceiling)
    )
    target = st.slider("Target accuracy", 0.55, 0.99, 0.90, 0.01)
    needed = samples_needed(p, rho, target)
    if needed is None:
        st.write(
            "Samples needed to reach %.2f: **unreachable** at this "
            "correlation (ceiling %.3f)." % (target, ceiling)
        )
    else:
        st.write("Samples needed to reach %.2f: **%d**" % (target, needed))


def tab_evaluation():
    st.header("Headline evaluation (stored results)")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "evaluation.json")
    if not os.path.exists(path):
        st.info("No stored results yet. Run: python -m vote_eff evaluate")
        return
    with open(path) as handle:
        evaluation = json.load(handle)
    agg = evaluation["aggregates"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Gain MAE, this tool", "%.4f" % agg["gain_mae_tool"])
    m2.metric("Gain MAE, naive independence", "%.4f" % agg["gain_mae_naive"])
    m3.metric(
        "Naive over-promise at high bias",
        "%+.4f" % agg["naive_overpromise_high_rho"],
    )
    st.bar_chart(
        {
            "gain prediction MAE": [agg["gain_mae_tool"], agg["gain_mae_naive"]],
        }
    )
    st.caption("Bars: this tool first, naive independence baseline second.")
    rows = []
    for scenario in evaluation["scenarios"]:
        rows.append(
            {
                "p": scenario["p"],
                "latent bias": scenario["rho_latent"],
                "rho true": round(scenario["rho_true"], 3),
                "rho hat": round(scenario["rho_hat"], 3),
                "n_eff true (32)": round(scenario["neff_true_32"], 2),
                "n_eff hat (32)": round(scenario["neff_hat_32"], 2),
                "rel. error %": round(100.0 * scenario["neff_rel_error_32"], 1),
            }
        )
    st.dataframe(rows, use_container_width=True)
    showcase = agg.get("showcase")
    if showcase:
        st.write(
            "Showcase: at p = %.2f with latent bias %.1f, 32 samples are "
            "worth %.2f independent votes; the audit estimates %.2f from a "
            "48 x 16 probe. Realised gain %.3f, tool prediction %.3f, naive "
            "promise %.3f."
            % (
                showcase["p"],
                showcase["rho_latent"],
                showcase["neff_true_32"],
                showcase["neff_hat_32"],
                showcase["realised_gain_32"],
                showcase["tool_pred_gain_32"],
                showcase["naive_pred_gain_32"],
            )
        )
    st.caption(
        "All votes are synthetic, from the seeded sampler with planted "
        "shared bias; ground-truth correlation is analytic. See "
        "results/evaluation_summary.md."
    )


tab1, tab2, tab3 = st.tabs(["Audit", "Prediction", "Evaluation"])
with tab1:
    tab_audit()
with tab2:
    tab_prediction()
with tab3:
    tab_evaluation()
