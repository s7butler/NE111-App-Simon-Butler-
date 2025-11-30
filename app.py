import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

st.set_page_config(page_title="Histogram Distribution Fitter", layout="wide")

# ----------------------------------------------------------
# 1. Title
# ----------------------------------------------------------
st.title("📊 Histogram Distribution Fitter")
st.write("Upload or enter data, then automatically or manually fit distributions.")

# ----------------------------------------------------------
# 2. Data Input Section
# ----------------------------------------------------------

st.header("1️⃣ Enter Data")

tab1, tab2 = st.tabs(["✍️ Manual Entry", "📁 Upload CSV"])

with tab1:
    text_data = st.text_area("Enter numbers separated by commas or spaces:")
    data = None
    if text_data.strip():
        try:
            data = np.array([float(x) for x in text_data.replace(",", " ").split()])
        except:
            st.error("Could not parse the data.")

with tab2:
    uploaded = st.file_uploader("Upload a CSV file with one column of numeric data:")
    if uploaded:
        df = pd.read_csv(uploaded)
        try:
            data = df.iloc[:,0].dropna().astype(float).values
        except:
            st.error("CSV must contain at least one numeric column.")

# If no data yet
if data is None:
    st.warning("Please enter or upload data to continue.")
    st.stop()

# ----------------------------------------------------------
# List of distributions (minimum 10)
# ----------------------------------------------------------
distribution_dict = {
    "Normal": stats.norm,
    "Exponential": stats.expon,
    "Gamma": stats.gamma,
    "Weibull": stats.weibull_min,
    "Lognormal": stats.lognorm,
    "Beta": stats.beta,
    "Chi-Square": stats.chi2,
    "Student-t": stats.t,
    "Laplace": stats.laplace,
    "Uniform": stats.uniform,
    "Logistic": stats.logistic,
}

# ----------------------------------------------------------
# 3. Distribution Selection
# ----------------------------------------------------------
st.header("2️⃣ Choose Distribution")

dist_name = st.selectbox("Select a distribution to fit:", list(distribution_dict.keys()))
dist = distribution_dict[dist_name]

# ----------------------------------------------------------
# Automatic fit
# ----------------------------------------------------------
params = dist.fit(data)

st.write(f"**Fitted parameters:** {params}")

# ----------------------------------------------------------
# Compute fit quality (mean error)
# ----------------------------------------------------------
xs = np.linspace(min(data), max(data), 200)
pdf_vals = dist.pdf(xs, *params)
hist_vals, bins = np.histogram(data, bins=20, density=True)
bin_centers = (bins[:-1] + bins[1:]) / 2

from sklearn.metrics import mean_absolute_error
fit_error = mean_absolute_error(hist_vals, dist.pdf(bin_centers, *params))

st.write(f"**Fit Mean Error:** {fit_error:.5f}")

# ----------------------------------------------------------
# 4. Manual Fitting Section
# ----------------------------------------------------------
st.header("3️⃣ Manual Fitting (Adjust Parameters)")

with st.expander("Show Manual Controls"):
    sliders = []
    for i, p in enumerate(params):
        sliders.append(st.slider(
            f"Parameter {i+1}",
            min_value=float(p - abs(p)*2 - 1),
            max_value=float(p + abs(p)*2 + 1),
            value=float(p),
            step=0.01
        ))
    manual_params = sliders

# ----------------------------------------------------------
# 5. Visualization
# ----------------------------------------------------------
st.header("4️⃣ Visualization")

fig, ax = plt.subplots(figsize=(8,5))

# Histogram
ax.hist(data, bins=20, density=True, alpha=0.5, color="lightblue", label="Data Histogram")

# Automatic fit curve
ax.plot(xs, pdf_vals, "r", lw=2, label=f"{dist_name} Fit")

# Manual fit curve
manual_pdf_vals = dist.pdf(xs, *manual_params)
ax.plot(xs, manual_pdf_vals, "g--", lw=2, label="Manual Fit")

ax.legend()
ax.set_xlabel("Value")
ax.set_ylabel("Density")
ax.set_title("Histogram + Fitted Distributions")

st.pyplot(fig)