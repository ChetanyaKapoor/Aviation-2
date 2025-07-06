
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, ConfusionMatrixDisplay)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

st.set_page_config(page_title="Aviation ML Workbench", layout="wide")
st.title("📈 Aviation – Classification Workbench")

# 1 ▸ Load data
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

DATA_PATH = "aviation final.csv"
df = load_data(DATA_PATH)
st.markdown(f"*Dataset loaded → **{df.shape[0]:,} rows** | **{df.shape[1]} columns***")

# 2 ▸ Sidebar
st.sidebar.header("⚙️ Setup")
label_col = st.sidebar.selectbox("Choose target column", options=df.columns, index=0)
test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", 0, value=42, step=1)
num_imputer = st.sidebar.selectbox("Numeric imputer", ["mean", "median"], 1)
cat_imputer = st.sidebar.selectbox("Categorical imputer", ["most_frequent", "constant"], 0)
run_button = st.sidebar.button("▶️ Run models")

# 3 ▸ EDA expander
with st.expander("🔍 Basic stats & 20‑plot EDA"):
    st.subheader("Null-percentage by column")
    st.write((df.isna().mean()*100).round(2).rename('% null').sort_values(ascending=False))
    st.subheader("Describe (numeric)")
    st.write(df.describe().T)

    NUM_PLOTS = 20
    plotted = 0
    for col in df.columns:
        if plotted >= NUM_PLOTS:
            break
        plt.figure()
        if pd.api.types.is_numeric_dtype(df[col]):
            sns.histplot(df[col].dropna(), kde=True)
        else:
            sns.countplot(y=df[col])
        plt.title(col)
        st.pyplot(plt.gcf())
        plt.close()
        plotted += 1

# Helper: preprocess
num_cols = df.select_dtypes(include="number").columns.drop(label_col)
cat_cols = df.select_dtypes(exclude="number").columns.drop(label_col)

preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy=num_imputer)),
        ("scaler", StandardScaler())
    ]), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy=("constant" if cat_imputer=="constant" else "most_frequent"),
                                  fill_value="missing")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols)
])

if run_button:
    X = df.drop(columns=[label_col])
    y = df[label_col]
    mask = y.notna()
    X, y = X.loc[mask], y.loc[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state)

    MODELS = {
        "K-Nearest Neighbours": (KNeighborsClassifier(n_neighbors=7), False),
        "Decision Tree": (DecisionTreeClassifier(random_state=random_state), True),
        "Random Forest": (RandomForestClassifier(n_estimators=250, random_state=random_state), True),
        "Gradient Boosting": (GradientBoostingClassifier(random_state=random_state), True)
    }

    results = []
    confusion_figs = []
    importance_figs = []

    for name, (est, has_imp) in MODELS.items():
        pipe = Pipeline([("prep", preprocess), ("clf", est)])
        pipe.fit(X_train, y_train)
        y_pred_train = pipe.predict(X_train)
        y_pred_test = pipe.predict(X_test)

        results.append({
            "Model": name,
            "Train Acc": accuracy_score(y_train, y_pred_train),
            "Test Acc": accuracy_score(y_test, y_pred_test),
            "Precision": precision_score(y_test, y_pred_test, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, y_pred_test, average="weighted", zero_division=0),
            "F1": f1_score(y_test, y_pred_test, average="weighted", zero_division=0)
        })

        fig_cm, ax_cm = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test, ax=ax_cm, cmap="Blues", colorbar=False)
        ax_cm.set_title(f"{name} – Confusion Matrix")
        confusion_figs.append((name, fig_cm))

        if has_imp:
            ohe_cols = pipe.named_steps["prep"].get_feature_names_out()
            importances = pipe.named_steps["clf"].feature_importances_
            imp_series = pd.Series(importances, index=ohe_cols).sort_values(ascending=False)[:20]
            fig_imp, ax_imp = plt.subplots(figsize=(5,4))
            sns.barplot(x=imp_series.values, y=imp_series.index, ax=ax_imp)
            ax_imp.set_title(f"{name} – Top 20 Features")
            importance_figs.append((name, fig_imp))

    st.subheader("📊 Performance Metrics")
    st.table(pd.DataFrame(results).set_index("Model").style.format("{:.3f}"))

    st.subheader("🔵 Confusion Matrices")
    cols_cm = st.columns(2)
    for i,(name, fig_cm) in enumerate(confusion_figs):
        cols_cm[i%2].markdown(f"**{name}**")
        cols_cm[i%2].pyplot(fig_cm)

    st.subheader("⭐ Feature Importances (tree models)")
    if importance_figs:
        cols_imp = st.columns(2)
        for i,(name, fig_imp) in enumerate(importance_figs):
            cols_imp[i%2].markdown(f"**{name}**")
            cols_imp[i%2].pyplot(fig_imp)
    else:
        st.info("K-NN has no intrinsic feature-importance.")
