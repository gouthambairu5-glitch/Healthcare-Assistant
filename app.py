import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- CONFIGURATION ---
st.set_page_config(
    page_title="AI Health Diagnosis Assistant",
    page_icon="🩺",
    layout="wide"
)

# --- MODEL AND DATA LOADING ---
@st.cache_data
def load_data_and_model():
    """Loads all necessary data and the trained model."""
    try:
        model = joblib.load("healthcare_ai_model.pkl")
        scaler = joblib.load("scaler.pkl")
        label_encoder = joblib.load("label_encoder.pkl")
        train_df = pd.read_csv("Training_Cleaned.csv")
        test_df = pd.read_csv("Testing_Cleaned.csv")
        description_df = pd.read_csv("symptom_Description_Updated.csv")
        precaution_df = pd.read_csv("symptom_Precaution_Updated.csv")

        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        ALL_SYMPTOMS = train_df.columns.drop('prognosis').tolist()

        return model, scaler, label_encoder, description_df, precaution_df, ALL_SYMPTOMS, combined_df
    except FileNotFoundError as e:
        st.error(f"Error loading files: {e}. Please make sure all required CSV and PKL files are in the same directory as the app.")
        return None, None, None, None, None, None, None

model, scaler, label_encoder, description_df, precaution_df, ALL_SYMPTOMS, combined_df = load_data_and_model()

# --- DISEASE CATEGORIES ---
DISEASE_CATEGORIES = {
    "Skin Conditions": {"diseases": ["Fungal infection", "Acne", "Psoriasis", "Impetigo"]},
    "Fever & Infections": {"diseases": ["Dengue", "Typhoid", "Common Cold", "Pneumonia", "Malaria", "Chicken pox"]},
    "Digestive & Liver Issues": {"diseases": ["GERD", "Chronic cholestasis", "Peptic ulcer diseae", "Jaundice",
                                               "Gastroenteritis", "Hepatitis A", "Hepatitis B", "Hepatitis C",
                                               "Hepatitis D", "Hepatitis E", "Alcoholic hepatitis"]},
    "Joint & Muscle Conditions": {"diseases": ["Arthritis", "Osteoarthristis", "Cervical spondylosis"]},
    "Neurological Conditions": {"diseases": ["(vertigo) Paroymsal  Positional Vertigo", "Migraine",
                                              "Paralysis (brain hemorrhage)"]}
}

# --- HELPER FUNCTIONS ---
@st.cache_data
def get_symptoms_for_diseases(disease_list, data_df):
    filtered_df = data_df[data_df['prognosis'].isin(disease_list)]
    symptoms = filtered_df.loc[:, (filtered_df != 0).any(axis=0)]
    return sorted([s for s in symptoms.columns if s != 'prognosis'])

def perform_prediction(selected_symptoms):
    if not selected_symptoms or model is None or scaler is None or label_encoder is None:
        return None, None, None

    input_features = [1 if symptom in selected_symptoms else 0 for symptom in ALL_SYMPTOMS]
    input_scaled = scaler.transform([input_features])

    probabilities = model.predict_proba(input_scaled)[0]
    top4_indices = np.argsort(probabilities)[-4:][::-1]
    top4_diseases = [(label_encoder.classes_[i], probabilities[i]) for i in top4_indices]

    top_disease_name = top4_diseases[0][0]
    desc_info = description_df[description_df['Disease'] == top_disease_name] if description_df is not None else pd.DataFrame()
    description = desc_info.iloc[0]['Disease Description'] if not desc_info.empty else "No description available."

    prec_info = precaution_df[precaution_df['Disease'] == top_disease_name] if precaution_df is not None else pd.DataFrame()
    precautions = []
    if not prec_info.empty:
        precaution_text = prec_info.iloc[0]['Precaution']
        if pd.notna(precaution_text):
            precautions = [p.strip() for p in precaution_text.split(';')]

    return top4_diseases, description, precautions

def render_diagnosis_page(category_name, symptoms_list):
    st.header(f"Diagnosis for: {category_name}")
    st.markdown("Select the symptoms you are experiencing from the list below.")

    selected_symptoms = st.multiselect("Symptoms", options=symptoms_list, help="You can select multiple symptoms.")

    if st.button(f"Predict Disease for {category_name}", key=category_name):
        if not selected_symptoms:
            st.warning("Please select at least one symptom.")
        else:
            with st.spinner("Analyzing your symptoms..."):
                predictions, description, precautions = perform_prediction(selected_symptoms)

                if predictions:
                    st.subheader("Prediction Results")
                    st.markdown("Based on your symptoms, here are the most likely conditions:")

                    for disease, probability in predictions:
                        st.write(f"**{disease}:** {probability*100:.2f}%")

                    st.markdown("---")
                    st.info(f"**Details for the most likely condition ({predictions[0][0]}):**\n\n{description}")

                    if precautions:
                        st.warning("**Recommended Precautions:**")
                        for p in precautions:
                            st.markdown(f"- {p}")
                    st.markdown("---")
                    st.markdown("*Disclaimer: This is an AI-powered prediction and not a substitute for professional medical advice. Please consult a doctor.*")
                else:
                    st.error("Could not make a prediction. Please try again.")

# --- MAIN APP LAYOUT ---
st.title("🩺 AI-Powered Health Diagnosis Assistant")
st.markdown("Navigate through different disease categories using the sidebar and select your symptoms to get a prediction.")

st.sidebar.title("Disease Categories")
page_options = list(DISEASE_CATEGORIES.keys())
selected_page = st.sidebar.radio("Select a category", page_options)

# --- SAFE FILE CHECK ---
if (
    model is not None
    and scaler is not None
    and label_encoder is not None
    and description_df is not None and not description_df.empty
    and precaution_df is not None and not precaution_df.empty
    and ALL_SYMPTOMS is not None and len(ALL_SYMPTOMS) > 0
    and combined_df is not None and not combined_df.empty
):
    diseases_in_category = DISEASE_CATEGORIES[selected_page]["diseases"]
    symptoms_for_category = get_symptoms_for_diseases(diseases_in_category, combined_df)
    render_diagnosis_page(selected_page, symptoms_for_category)
else:
    st.error("Application cannot start because one or more essential files could not be loaded or are empty.")
