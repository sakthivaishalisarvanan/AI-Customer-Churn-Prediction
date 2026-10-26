import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

MODEL_PATH = "trained_model.pkl"
PREP_PATH = "preprocessing.pkl"
TARGET_COL = "churn"
DROP_COLS = ["customer_id", "signup_date"]

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\sakth\Documents\AI-customer churn\customer_churn_1M[1].csv")
    return df

df = load_data()

# ---------------- SIDEBAR ----------------
menu = st.sidebar.selectbox(
    "Menu",
    ["Dashboard", "Train Model", "Prediction", "Logout"]
)

# ---------------- HELPER: BUILD CLEAN TRAINING TABLE ----------------
def build_training_table(raw_df):
    """
    Cleans the raw dataframe and returns:
    data          -> fully numeric dataframe ready for the model
    encoders      -> dict of {column: LabelEncoder} for categorical columns
    num_fill      -> dict of {column: fill_value} used for missing numeric values
    feature_cols  -> list of feature column names in the exact order used for training
    cat_cols      -> list of categorical column names
    num_cols      -> list of numeric column names (features only, no target)
    """
    data = raw_df.copy()

    # Drop identifier / non-predictive columns
    data.drop(columns=DROP_COLS, inplace=True, errors="ignore")

    # Make sure target is numeric 0/1 and has no missing values
    data[TARGET_COL] = pd.to_numeric(data[TARGET_COL], errors="coerce")
    data.dropna(subset=[TARGET_COL], inplace=True)
    data[TARGET_COL] = data[TARGET_COL].astype(int)

    cat_cols = [c for c in data.columns
                if c != TARGET_COL and (data[c].dtype == "object" or str(data[c].dtype) == "string")]
    num_cols = [c for c in data.columns if c != TARGET_COL and c not in cat_cols]

    # Force numeric columns to numeric, coercing bad strings (e.g. blanks) to NaN
    for col in num_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Fill missing numeric values with the median, save the value used
    num_fill = {}
    for col in num_cols:
        fill_val = data[col].median()
        num_fill[col] = fill_val
        data[col] = data[col].fillna(fill_val)

    # Fill missing categorical values with the mode, then label-encode
    encoders = {}
    for col in cat_cols:
        mode_val = data[col].mode(dropna=True)
        mode_val = mode_val.iloc[0] if len(mode_val) else "Unknown"
        data[col] = data[col].fillna(mode_val).astype(str)

        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        encoders[col] = le

    feature_cols = [c for c in data.columns if c != TARGET_COL]

    return data, encoders, num_fill, feature_cols, cat_cols, num_cols


# ---------------- DASHBOARD ----------------
if menu == "Dashboard":
    st.title("AI Based Customer Churn Prediction")
    st.write("Dataset Preview")
    st.dataframe(df.head())

    st.write("Columns")
    st.write(df.columns.tolist())

    st.write("Churn distribution")
    st.bar_chart(df[TARGET_COL].value_counts())

# ---------------- TRAIN MODEL ----------------
elif menu == "Train Model":
    st.title("Train Model")

    data, encoders, num_fill, feature_cols, cat_cols, num_cols = build_training_table(df)

    X = data[feature_cols]
    y = data[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model_name = st.selectbox(
        "Select Model",
        ["Random Forest", "XGBoost", "Decision Tree"]
    )

    if st.button("Train"):
        if model_name == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_name == "XGBoost":
            model = XGBClassifier(eval_metric="logloss", random_state=42)
        elif model_name == "Decision Tree":
            model = DecisionTreeClassifier(random_state=42)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        st.success(f"{model_name} Accuracy: {acc*100:.2f}%")

        # Save model
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)

        # Save everything needed to transform raw user input the same way
        preprocessing = {
            "encoders": encoders,
            "num_fill": num_fill,
            "feature_cols": feature_cols,
            "cat_cols": cat_cols,
            "num_cols": num_cols,
            "raw_df_for_ranges": df,  # used to build sensible input widgets
        }
        with open(PREP_PATH, "wb") as f:
            pickle.dump(preprocessing, f)

        st.info("Model and preprocessing pipeline saved. Go to 'Prediction' to test it.")

# ---------------- PREDICTION ----------------
elif menu == "Prediction":
    st.title("Prediction")

    if not (os.path.exists(MODEL_PATH) and os.path.exists(PREP_PATH)):
        st.warning("Train the model first on the 'Train Model' page.")
    else:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(PREP_PATH, "rb") as f:
            prep = pickle.load(f)

        encoders = prep["encoders"]
        num_fill = prep["num_fill"]
        feature_cols = prep["feature_cols"]
        cat_cols = prep["cat_cols"]
        num_cols = prep["num_cols"]
        raw_df = prep["raw_df_for_ranges"]

        st.write("Enter customer details:")
        user_input = {}

        col1, col2 = st.columns(2)
        cols_cycle = [col1, col2]

        # Categorical inputs -> dropdown of the original text values
        for i, col in enumerate(cat_cols):
            options = list(encoders[col].classes_)
            with cols_cycle[i % 2]:
                user_input[col] = st.selectbox(col, options)

        # Numeric inputs -> number box, defaulted/ranged from training data
        for i, col in enumerate(num_cols):
            series = pd.to_numeric(raw_df[col], errors="coerce")
            min_v = float(series.min()) if series.notna().any() else 0.0
            max_v = float(series.max()) if series.notna().any() else 100.0
            default_v = float(num_fill[col])
            with cols_cycle[(i + len(cat_cols)) % 2]:
                user_input[col] = st.number_input(
                    col, min_value=min_v, max_value=max_v, value=default_v
                )

        if st.button("Predict"):
            # Build a single-row dataframe in the exact training column order
            row = {}
            for col in feature_cols:
                if col in cat_cols:
                    row[col] = encoders[col].transform([user_input[col]])[0]
                else:
                    row[col] = user_input[col]

            input_df = pd.DataFrame([row], columns=feature_cols)
            prediction = model.predict(input_df)[0]

            proba = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_df)[0][1]

            if prediction == 1:
                st.error("Customer Will Churn" + (f" (probability: {proba:.2%})" if proba is not None else ""))
            else:
                st.success("Customer Will Not Churn" + (f" (probability of churn: {proba:.2%})" if proba is not None else ""))

# ---------------- LOGOUT ----------------
elif menu == "Logout":
    st.title("Logout")
    st.success("Logged out successfully!")