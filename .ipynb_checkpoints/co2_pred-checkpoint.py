import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from PIL import Image


# ✅ Set dynamic layout
st.set_page_config(layout="wide")  # Enables responsive width

# ✅ Load and Display the Header Image
header_image_path = "app/header.png"
try:
    image = Image.open(header_image_path)
    st.image(image, use_container_width=True)  # Full-width image # Alt: , caption="An AI-generated image of a sports car surrounded by smoke"
except FileNotFoundError:
    st.warning("⚠️ Header image not found! Ensure 'header.png' is in the directory.")

# ✅ Load the trained Random Forest model correctly
@st.cache_resource
def load_model():
    try:
        model = joblib.load("app/random_forest_compressed.pkl")
        if not hasattr(model, "predict"):
            st.error("❌ Loaded model is not a valid Random Forest model!")
            return None
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

@st.cache_resource
def load_feature_columns():
    try:
        return joblib.load("app/feature_columns.pkl")
    except Exception as e:
        st.error(f"❌ Error loading feature columns: {e}")
        return None

# Load Model & Features
rf_model = load_model()
feature_columns = load_feature_columns()
if rf_model is None or feature_columns is None:
    st.stop()

# ✅ Load Sample Data & Residuals
try:
    sample_data = pd.read_csv("data/sample_data.csv")  
    sample_co2_data = pd.read_csv("data/sample_co2.csv")  
    actual = pd.read_csv("data/actual_rf.csv").values.flatten()
    predicted_rf = pd.read_csv("data/predicted_rf.csv").values.flatten()
    residuals = pd.read_csv("data/residuals_rf.csv").values.flatten()
    std_residuals = np.std(residuals, ddof=1)
except FileNotFoundError as e:
    st.error(f"❌ Error loading dataset: {e}")
    residuals = None
    std_residuals = None
    
# ✅ Define Confidence Interval (95%)
z_score = 1.96  # For 95% confidence interval
confidence_margin = z_score * std_residuals if std_residuals is not None else 0

# ✅ Residual Plot Function
def plot_residuals(show_prediction=False, predicted_value=None, lower_bound=None, upper_bound=None):
    plt.style.use("dark_background") # Apply dark theme to match the app interface
    plt.figure(figsize=(6, 4))
    # KDE plots for actual and predicted emissions
    ax = sns.kdeplot(actual, color="#1E90FF", label="Actual CO₂ Emission Rate", bw_adjust=0.3, linewidth=6)  # Bright Blue- Actual
    sns.kdeplot(predicted_rf, color="red", label="Random Forest Predictions", ax=ax, bw_adjust=0.3, linewidth=2)  # Bright Orange- RF Predictions
    # Customize grid and labels for dark theme
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.xlabel("CO₂ Emission Rate (g/mi)", fontsize=12, color='white')
    plt.ylabel("Proportion of Cars (×10⁻³)", fontsize=12, color='white')  # Adjusted label
    # Scale y-axis to remove leading zeros
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y * 1000:.0f}"))
    plt.title("Predicted vs. Actual CO₂ Emission Rates Distribution", fontsize=14, color='white')
    if show_prediction and predicted_value is not None:
        plt.axvline(x=predicted_value, color="#FFA500", linestyle='solid', linewidth=1, label="Predicted Emission Rate") # Add prediction result as a text annotation
        plt.axvspan(lower_bound, upper_bound, color="green", alpha=1, label="95% Confidence Interval") # Add confidence interval as a vertical band

    plt.legend(facecolor='black', edgecolor='white', fontsize=10) # Customize legend
    st.pyplot(plt)    
    
    
# ✅ TRANSMISSION & FUEL TYPE MAPPINGS
TRANSMISSION_MAPPING = {
    "Automatic (A)": "A",
    "Automatic with Select Shift (AS)": "AS",
    "Automated Manual (AM)": "AM",
    "Automated Continuous Variable (AV)": "AV",
    "Manual (M)": "M"
}

FUEL_TYPES_MAPPING = {
    "Gasoline- Regular (Z)": "Z",
    "Gasoline- Premium (X)": "X",
    "Diesel (D)": "D",
    "E85 (E)": "E",
    "Natural Gas (N)": "N"
}

# ✅ Store input values in session state
if "inputs" not in st.session_state:
    st.session_state.inputs = {}
    
# ✅ Create containers for structured layout
header_container = st.container()  # Header
middle_container = st.container()  # Holds Intro, Inputs-Samples-Predict
project_container = st.container()  # Project Description
info_container = st.container()  # info

# ✅ HEADER SECTION
with header_container:
    st.markdown("<h1 style='text-align: center;'>Vehicle CO₂ Emission Rate Predictor 🚗💨</h1>", unsafe_allow_html=True)

# ✅ MIDDLE SECTION: Holds three sections in a **flexible layout**
with middle_container:
    # 🚀 Check screen width for adaptive layout
    if st.session_state.get("screen_width", 1200) > 768:
        col1, col2, col3 = st.columns([0.3, 0.35, 0.35])  # Side-by-side layout for wider screens
    else:
        col1, col2, col3 = st.columns(1)  # Stacked layout for smaller screens

# ✅ INTRODUCTION SECTION
with col1:
    st.subheader("📌 Introduction")
    st.markdown("""
    This app predicts use-phase **CO₂ emission rates (g/mi)** for Internal Combustion Engine Vehicles (ICEVs) based on their specifications using a **Random Forest ML Model** with 99.56% accuracy.

    **How to use this app:**  
    1️⃣ **Choose an input method:** Select "Customize Input" to enter vehicle specs or pick a sample vehicle.  
    2️⃣ **Enter details (if Customize input):** Adjust specs like engine size, transmission type, and fuel type. Inputs are **disabled** for samples but show their values.  
    3️⃣ **Click "Apply Vehicle Specification" button (if "Customize Input") to update inputs.  
    4️⃣ **Click "Predict CO₂ Emission Rate"** to generate a prediction with a **95% confidence interval**.  
    5️⃣ **Review results: Examine the predicted CO₂ emission rate and confidence interval.  
    6️⃣ **Compare with actual emission rates: If a sample is selected, the actual CO₂ value appears with a % difference analysis.  
    7️⃣ **Check distribution plot on the right. It updates after prediction, showing the model estimation, confidence interval, and how it compares to the overall distribution.  

    Built in ***Python*** and ***Streamlit***, using a 27k data records from the **Natural Resources Canada**. For more details, scroll down.
    """)

    # ✅ INPUTS, SAMPLES, AND PREDICTION SECTION
    with col2:
        st.subheader("🚘 Vehicle Specifications")
        input_options = ["Customize Input", "2018 Toyota Corolla", "2015 Subaru Legacy", "2000 Chevrolet Camaro"]
        selected_option = st.selectbox("Choose Input Type", input_options)

        # Load Selected Sample Data
        if selected_option == "Customize Input":
            selected_sample = None
        else:
            sample_index = input_options.index(selected_option) - 1  # Adjust index to match DataFrame
            selected_sample = sample_data.iloc[sample_index].to_dict()
            st.session_state.inputs.update(selected_sample)


        # ✅ User Inputs
        disabled = selected_sample is not None  # Disable input fields if a sample is selected
        
        with st.form("addition_form"):
            col21, col22, col23 = st.columns(3)
            st.session_state.inputs["Model year"] = col21.number_input("Model Year", min_value=1995, max_value=2030,
                                           value=int(selected_sample["Model year"]) if selected_sample else 2026,
                                           disabled=disabled)
            st.session_state.inputs["Engine size (L)"] = col22.number_input("Engine Size (L)", min_value=0.8, max_value=10.0, step=0.1,
                                            value=float(selected_sample["Engine size (L)"]) if selected_sample else 2.0,
                                            disabled=disabled)
            st.session_state.inputs["Cylinders"] = col23.number_input("Number of Cylinders", min_value=2, max_value=16, step=1,
                                          value=int(selected_sample["Cylinders"]) if selected_sample else 4,
                                          disabled=disabled)

            col24, col25, col26 = st.columns(3)
            st.session_state.inputs["Number of gears-speeds"] = col24.number_input("Number of Gears-Speeds", min_value=3, max_value=10, step=1,
                                      value=int(selected_sample["Number of gears-speeds"]) if selected_sample else 6,
                                      disabled=disabled)
            st.session_state.inputs["City (mpg)"] = col25.number_input("City MPG", min_value=5.0, max_value=45.0, step=0.1,
                                         value=float(selected_sample["City (mpg)"]) if selected_sample else 25.0,
                                         disabled=disabled)
            st.session_state.inputs["Highway (mpg)"] = col26.number_input("Highway MPG", min_value=5.0, max_value=55.0, step=0.1,
                                            value=float(selected_sample["Highway (mpg)"]) if selected_sample else 32.0,
                                            disabled=disabled)

            col27, col28 = st.columns(2)
            st.session_state.inputs["Transmission type"] = col27.selectbox("Transmission Type", list(TRANSMISSION_MAPPING.keys()),
                                               index=list(TRANSMISSION_MAPPING.values()).index(selected_sample["Transmission type"])
                                               if selected_sample else 0, disabled=disabled)
            st.session_state.inputs["Fuel type"] = col28.selectbox("Fuel Type", list(FUEL_TYPES_MAPPING.keys()),
                                       index=list(FUEL_TYPES_MAPPING.values()).index(selected_sample["Fuel type"])
                                       if selected_sample else 0, disabled=disabled)
            # ✅ Submit button
            submitted = st.form_submit_button("Apply Vehicle Specifications",  disabled=disabled)
        
        # ✅ Prediction Button (Custom Styling)
        st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: darkorange;
            color: black;
            font-size: 18px;
            font-weight: bold;
            width: 100%;
            border-radius: 10px;
        }
        div.stButton > button:first-child:hover {
            background-color: #cc7000;
        }
        </style>
        """, unsafe_allow_html=True)
        
        if st.button("Predict CO₂ Emission Rate"):
            # Step 1: Create DataFrame for User Input
            user_input = pd.DataFrame([st.session_state.inputs])
            user_input_mapped = pd.DataFrame([{
                "Model year": user_input["Model year"],
                "Engine size (L)": user_input["Engine size (L)"],
                "Cylinders": user_input["Cylinders"],
                "Transmission type": TRANSMISSION_MAPPING[user_input["Transmission type"].iloc[0]],
                "Number of gears-speeds": user_input["Number of gears-speeds"],
                "Fuel type": FUEL_TYPES_MAPPING[user_input["Fuel type"].iloc[0]],
                "City (mpg)": user_input["City (mpg)"],
                "Highway (mpg)": user_input["Highway (mpg)"]
            }])

            # Step 2: Apply One-Hot Encoding to Categorical Features
            user_input_encoded = pd.get_dummies(user_input_mapped, columns=["Transmission type", "Fuel type"])

            # Step 3: Ensure Feature Alignment with `feature_columns`
            missing_cols = set(feature_columns) - set(user_input_encoded.columns)
            for col in missing_cols:
                user_input_encoded[col] = 0 # Add missing categorical variables with 0

            # Step 4: Drop Extra Columns That Were Not in `df_encoded`
            extra_cols = set(user_input_encoded.columns) - set(feature_columns)
            user_input_encoded = user_input_encoded.drop(columns=extra_cols, errors="ignore")
                
            # Step 5: Reorder Columns to Match Training Data (`df_encoded`)
            user_input_encoded = user_input_encoded[feature_columns]

            # Combine predictions (90% weight to SVR, 10% weight to RF)
            st.session_state["predicted_value"] = rf_model.predict(user_input_encoded)[0]
            st.session_state["lower_bound"] = st.session_state["predicted_value"] - confidence_margin
            st.session_state["upper_bound"] = st.session_state["predicted_value"] + confidence_margin
            
            # ✅ Step 7: Show Actual CO₂ Only If a Sample Was Selected & No Customization
            if selected_sample and not sample_co2_data.empty:
                actual_value = sample_co2_data.iloc[sample_index]["CO2 emissions (g/mi)"]

                # ✅ Calculate Difference & Generate Justification Message
                difference = abs(st.session_state["predicted_value"] - actual_value)
                percentage_diff = (difference / actual_value) * 100

                if percentage_diff < 1:
                    justification_text = "✅ **Excellent match!**" 
                elif percentage_diff < 5:
                    justification_text = "🔹 **Good match!**"
                else:
                    justification_text = "⚠️ **Higher deviation detected.**"

                # ✅ Display Results
                st.success(f"**Predicted CO₂ Emission Rate:** {st.session_state['predicted_value']:.2f} g/mi")
                if std_residuals is not None:
                    st.write(f"95% Confidence Interval: [{st.session_state['lower_bound']:.2f}, {st.session_state['upper_bound']:.2f}] g/mi")
                st.info(f"**Actual CO₂ Emission Rate:** {actual_value:.2f} g/mi")
                st.write(f"**Difference:** {difference:.2f} g/mi ({percentage_diff:.2f}%)")
                st.write(justification_text)                

            else:
                st.success(f"Predicted CO₂ Emission Rate: {st.session_state['predicted_value']:.2f} g/mi")
                if std_residuals is not None:
                        st.write(f"95% Confidence Interval: [{st.session_state['lower_bound']:.2f}, {st.session_state['upper_bound']:.2f}] g/mi")

                    

    # ✅ Visualization Section (Always Show Plot)
    with col3:
        st.subheader("📈 Distributions Plot")
        plot_residuals(
            show_prediction="predicted_value" in st.session_state,
            predicted_value=st.session_state.get("predicted_value"),
            lower_bound=st.session_state.get("lower_bound"),
            upper_bound=st.session_state.get("upper_bound")
        )
        st.write("The plot shows the distributions of actual and predicted CO₂ emission rates. The blue line represents the actual values and the red line shows the predicted  emission rate. The model (red) smoothly follows the actual distribution (blue), even in high and low emission rate ranges. The yellow thin vertical line represents the predicted value for the input, along with its confidence interval band (green).")

            
# ✅ PROJECT OVERVIEW SECTION
with project_container:
    st.markdown("---")
    st.subheader("📌 Project Description")
    col1, col2, col3 = st.columns([0.3, 0.35, 0.35])

    with col1:
        st.markdown("""
        This app stems from a broader machine-learning study on predicting use-phase CO₂ emission rates of Internal Combustion Engine Vehicles (ICEVs).
        
        The model uses data from 12 datasets by Natural Resources Canada, comprising 28,384 records for model years 1995-2025. Rigorous data preparation included handling missing data, feature engineering, variable selection, outlier detection, and categorical encoding. The prepared dataset had 27,742 rows and 17 columns.       
        
        **Exploratory Data Analysis (EDA)** was conducted to identify relationships between CO₂ emission rates and influencing features. This analysis included **univariate, bivariate, and multivariate visualizations**, which helped guide feature selection and modeling decisions.
        """)

    with col2:
        st.markdown("""
        Three machine learning approaches were tested: Multiple Ridge Regression, 2nd-degree Polynomial Ridge Regression, and Random Forest, with hyperparameter optimization. The prepared dataset was split 80/20 for training/testing. The random forest regressor (n_estimators=200, max_depth=20) performed best, showing Cross-Val R² = 0.9956.
        
        For deployment, the compressed model was exported using joblib and integrated into the Streamlit-based interactive interface.
        
        This project expands on an ***IBM Data Science Professional Certificate*** exercise, where a small subset of the 1995-2004 dataset (~1,000 records) was used to build a multiple linear regression model based on selected numerical variables. This work significantly expanded the dataset, included categorical variables, and employed new ML methods for improved predictive accuracy. 
        """)

    with col3:
        st.markdown("""
        Throughout this project, **prompt engineering** with **ChatGPT** was utilized for code development, optimization, image generation (header), and content structuring.
        
        If you’re interested, check out the GitHub repository for the project Markdown and the app code. Feel free to reach out if you have any questions or comments—I’d love to hear your thoughts! You can find my contact information below.

        **📂 Data Sources:** [Natural Resources Canada](https://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64)  
        **📂 App Code & Jupyter Notebook:** [GitHub Repo](https://github.com/hroshan/CO2_Emissions_Prediction)  
        **⚙️ Data Preparation:** Pandas, NumPy  
        **📈 Visualization:** Matplotlib, Seaborn  
        **🤖 Machine Learning:** Scikit-learn  
        **🖥️ App Development:** Streamlit, joblib
        
            """)
                    
# ✅ Info SECTION
with info_container:
    st.markdown("---")
    st.subheader("📌 About")
    st.markdown(""" 

    👨‍💻 **Author:** Hasan Roshan – Sustainability Analyst, Ph.D. in Environmental & Natural Resource Sciences  
    🔗 **LinkedIn:** [linkedin.com/in/hasanroshan](https://linkedin.com/in/hasanroshan)  
    📢 **Disclaimer:** This app is intended for educational purpose only.      
    🗓️ **Last Updated:** May 6, 2025
    """)
