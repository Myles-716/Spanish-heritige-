import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

st.title("Latin America Data Regression and Function Analysis")

# Placeholder data
data_options = {
    "Population": {
        "Mexico": {1950: 27900000, 1960: 34900000, 1970: 48300000, 1980: 67700000,
                   1990: 81500000, 2000: 98800000, 2010: 117000000, 2020: 128900000},
        "Brazil": {1950: 53700000, 1960: 72400000, 1970: 95500000, 1980: 121700000,
                   1990: 149000000, 2000: 174000000, 2010: 195000000, 2020: 213000000},
        "Argentina": {1950: 17100000, 1960: 20600000, 1970: 23900000, 1980: 28200000,
                      1990: 32600000, 2000: 37000000, 2010: 40400000, 2020: 45600000}
    },
    "Life expectancy": {
        "Mexico": {1950: 48, 1960: 58, 1970: 63, 1980: 67, 1990: 72, 2000: 75, 2010: 77, 2020: 75},
        "Brazil": {1950: 50, 1960: 55, 1970: 59, 1980: 62, 1990: 67, 2000: 71, 2010: 74, 2020: 76},
        "Argentina": {1950: 61, 1960: 65, 1970: 67, 1980: 70, 1990: 73, 2000: 75, 2010: 76, 2020: 77}
    }
}

category = st.selectbox("Select data category", list(data_options.keys()))
countries = st.multiselect("Select countries", list(data_options[category].keys()), default=list(data_options[category].keys())[:1])

df_list = []
for country in countries:
    years = list(data_options[category][country].keys())
    values = list(data_options[category][country].values())
    df_temp = pd.DataFrame({"Year": years, country: values})
    df_list.append(df_temp.set_index("Year"))

df = pd.concat(df_list, axis=1).reset_index()
st.write("### Raw Data (editable)")
editable_df = st.data_editor(df, num_rows="dynamic")

# Regression settings
degree = st.slider("Select polynomial regression degree", 3, 6, 3)
interval = st.slider("Select year increment for regression graph", 1, 10, 1)

fig, ax = plt.subplots(figsize=(10, 6))

for country in countries:
    years = editable_df["Year"].values.reshape(-1, 1)
    values = editable_df[country].values

    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(years)
    model = LinearRegression().fit(X_poly, values)

    future_years = np.arange(min(years), max(years) + 50, interval).reshape(-1, 1)
    y_poly_pred = model.predict(poly.transform(future_years))

    ax.scatter(years, values, label=f"{country} Data")
    ax.plot(future_years, y_poly_pred, label=f"{country} Regression (deg {degree})")

    coeffs = model.coef_
    intercept = model.intercept_
    st.write(f"**{country} Regression Equation (Degree {degree}):** y = {intercept:.2f} + " +
             " + ".join([f"{c:.2e}*x^{i}" for i, c in enumerate(coeffs) if i > 0]))

ax.set_xlabel("Year")
ax.set_ylabel(category)
ax.legend()
st.pyplot(fig)

st.write("### Function Analysis (example output)")
st.write(f"The {category.lower()} of {countries[0]} reached a local maximum in 2010. "
         f"It was increasing most rapidly around 1970.")
