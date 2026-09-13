from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


st.set_page_config(page_title="Car Price Prediction", page_icon="🚗")


@st.cache_resource
def train_model():
    path = Path("cardekho.csv")
    if not path.exists():
        path = Path("data/cardekho.csv")
    data = pd.read_csv(path).drop_duplicates().copy()
    data.columns = (data.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
                    .str.replace("(", "", regex=False).str.replace(")", "", regex=False)
                    .str.replace("/", "_", regex=False))
    for column in ["mileagekm_ltr_kg", "engine", "max_power"]:
        data[column] = pd.to_numeric(data[column].astype(str).str.extract(r"([-+]?\d*\.?\d+)")[0], errors="coerce")
    data["seats"] = pd.to_numeric(data["seats"], errors="coerce")
    data.loc[data["mileagekm_ltr_kg"].eq(0), "mileagekm_ltr_kg"] = np.nan
    parts = data["name"].astype(str).str.split(n=1, expand=True)
    data["company"] = parts[0]
    data["model"] = parts[1].fillna("Unknown")
    data["car_age"] = 2026 - data["year"]
    data["km_per_year"] = data["km_driven"] / data["car_age"].clip(lower=1)
    data = data.drop(columns=["name", "year"])
    X = data.drop(columns="selling_price")
    y = data["selling_price"]
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical = X.select_dtypes(include=np.number).columns.tolist()
    prep = ColumnTransformer([
        ("num", Pipeline([("scale", StandardScaler()), ("impute", KNNImputer())]), numerical),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
    ])
    model = Pipeline([("preprocess", prep), ("model", GradientBoostingRegressor(random_state=42))])
    model.fit(X, y)
    return model, data, categorical, numerical


st.title("CarDekho Used-Car Price Prediction")
st.caption("Enter a vehicle profile to estimate its selling price.")
model, data, categorical, numerical = train_model()

with st.form("prediction"):
    row = {}
    for column in categorical:
        row[column] = st.selectbox(column.replace("_", " ").title(), sorted(data[column].dropna().astype(str).unique()))
    defaults = {"km_driven": 50000.0, "mileagekm_ltr_kg": 18.0, "engine": 1200.0,
                "max_power": 85.0, "seats": 5.0, "car_age": 5.0, "km_per_year": 10000.0}
    for column in numerical:
        row[column] = st.number_input(column.replace("_", " ").title(), value=float(defaults.get(column, data[column].median())))
    submitted = st.form_submit_button("Predict price")

if submitted:
    prediction = max(0, float(model.predict(pd.DataFrame([row]))[0]))
    st.metric("Estimated selling price", f"₹{prediction:,.0f}")

