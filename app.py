# app.py
import streamlit as st
import pandas as pd

from mxmh_model import get_ui_schema, predict_from_user_dict, get_label_map

st.set_page_config(page_title = "Music Effects Predictor", page_icon = "🎧", layout  = "centered")

with st.spinner("Loading model & building form..."):
    schema = get_ui_schema()
    label_map = get_label_map()
    feature_cols = schema["feature_cols"]
    categorical = schema["categorical"]
    numerical = schema["numerical"]

# UI Header
st.title("Music Effects Predictor")
st.markdown("Answer all questions below to predict music effects on your mental health:")

# Tiny style touch to mimic a blue sections title
st.markdown(
    '''
    <style>
      .section-title {font-size:1.05rem; font-weight:600; color:#0b66ff; margin-top:0.75rem;}
    </style>
    ''',
    unsafe_allow_html = True
)
st.markdown('<div class="section-title">Basic Information</div>', unsafe_allow_html = True)

# Defaults so all required features are filled for the model
defaults = {}
for col in feature_cols:
    if col in categorical:
        defaults[col] = categorical[col][0] if categorical[col] else ""
    elif col in numerical:
        defaults[col] = float(numerical[col]["median"])
    else:
        defaults[col] = ""

# Helper to find actual column names in the dataset (case-insensitive)
def find_col(candidates):
    low = [c.lower() for c in feature_cols]
    for cand in candidates:
        if cand.lower() in low:
            return feature_cols[low.index(cand.lower())]
    return None

age_col = find_col(["Age"])
hours_col = find_col(["Hours per day", "Hours_per_day", "Hours Per Day"])
genre_col = find_col(["Fav genre", "Favorite genre", "Favourite genre"])
work_col =  find_col(["While working", "While Working"])

with st.form("user_input_form", clear_on_submit = False):
    # Age
    if age_col:
        age_val = st.number_input(
            "How old are you?", min_value = 0, max_value = 120, 
            value = int(defaults.get(age_col, 25))
        )
    # Hours per day
    if hours_col:
        hours_val = st.slider(
            "How many hours per day do you listen to music?", 
            min_value = 0.0, max_value = 24.0, 
            value = float(defaults.get(hours_col, 2.0)), step = 0.5
        )
    # Favorite genre
    if genre_col:
        genre_choices = categorical.get(genre_col, [])
        genre_val = st.selectbox(
            "What is your favorite genre?",
            options = genre_choices or ["(no choices found)"]
        )
    # While working
    if work_col:
        work_choices = categorical.get(work_col, ["Yes", "No"]) or ["Yes", "No"]
        default_idx = 0
        if "Yes" in work_choices:
            default_idx = work_choices.index("Yes")
        work_val = st.radio(
            "Do you listen to music while working?", 
            options = work_choices, index = default_idx
        )

    # Advanced expander
    with st.expander("Advanced: edit all features (optional)"):
        advanced = {}
        for col in feature_cols:
            if col in {age_col, hours_col, genre_col, work_col}:
                continue
            if col in categorical:
                advanced[col] = st.selectbox(col, options = categorical[col], key = f"adv_{col}")
            elif col in numerical:
                advanced[col] = st.number_input(col, value = float(defaults[col]), key = f"adv_{col}")
            else:
                advanced[col] = st.text_input(col, value = str(defaults[col]), key = f"adv_{col}")

    submitted = st.form_submit_button("Predict")

if submitted:
    # Build inputs dict covering ALL features
    user_inputs = defaults.copy()
    if age_col: user_inputs[age_col] = age_val
    if hours_col: user_inputs[hours_col] = hours_val
    if genre_col: user_inputs[genre_col] = genre_val
    if work_col: user_inputs[work_col] = work_val
    if "advanced" in locals():
        user_inputs.update(advanced)

    try:
        out = predict_from_user_dict(user_inputs)
        st.success(f"Prediction: **{out['pred_label']}**")
        if out.get("proba"):
            st.write("Class probabilities:")
            st.write(pd.DataFrame([out["proba"]]))
    except Exception as e:
        st.error(f"Prediction failed: {e}")


