import streamlit as st 
import pandas as pd
import numpy as np 
import joblib
import pickle


# Load model, scaler, and feature order 
model = joblib.load('rf_model.pkl')

with open('scaler.pkl', 'rb') as f:
  scaler = pickle.load(f)

with open('feature_columns.pkl', 'rb') as f:
  feature_columns = pickle.load(f)

st.set_page_config(page_title="Forest Cover Type Prediction", layout="centered")
st.title("🌳🌳 Forest Cover Type Prediction🌳🌳")
st.write("Enter the cartographic details below to predict the feature cover type")

# Continuous Features 
st.subheader("Terrain & Distance Features")
elevation = st.number_input("Elevation (meter)", min_value=0, max_value=4000, value=2800)
aspect = st.number_input("Aspect (degrees, 0-360)", min_value=0, max_value=360, value=150)
slope = st.number_input("Slope (degrees)", min_value=0, max_value=90, value=15)
h_dist_hydro = st.number_input("Horizontal Distance To Hydrology (m)", min_value=0, value=1500)
v_dist_hydro = st.number_input("Vertical Distance To Hydrology (m)", value=30)
hdist_road = st.number_input("Horizontal Distance To Roadways (m)", min_value=0, value=1500)
hillshade_9am = st.slider("Hillshade at 9am (0-255)", 0, 255, 210)
hillshade_noon = st.slider("Hillshade at Noon (0-255)", 0, 255, 220)
hillshade_3pm = st.slider("Hillshade at 3pm (0-255)", 0, 255, 140)
h_dist_fire = st.number_input("Horizontal Distance To Fire Points (m)", min_value=0, value=1200)

# Wilderness Area (pick one, 1 of 4)
st.subheader("Wilderness Area")
wilderness = st.selectbox("Select Wilderness Area", [1,2,3,4])

# Soil Type (pick one , 1 of 40)
st.subheader("Soil Type")
soil_type = st.selectbox("Select Soil Type", list(range(1,41)))

# Build input row matching training feature order 
if st.button("Predict Cover Type"):
  input_dict = dict.fromkeys(feature_columns,0)

  input_dict['Elevation'] = elevation 
  input_dict['Aspect'] = aspect
  input_dict['Slope'] = slope
  input_dict['Horizontal_Distance_To_Hydrology'] = h_dist_hydro
  input_dict['Vertical_Distance_To_Hydrology'] = v_dist_hydro
  input_dict['Horizontal_Distance_To_Roadways'] = hdist_road  
  input_dict['Hillshade_9am'] = hillshade_9am
  input_dict['Hillshade_Noon'] = hillshade_noon
  input_dict['Hillshade_3pm'] = hillshade_3pm
  input_dict['Horizontal_Distance_To_Fire_Points'] = h_dist_fire
  input_dict[f'Wilderness_Area{wilderness}'] = 1
  input_dict[f'Soil_Type{soil_type}'] = 1

  input_df = pd.DataFrame([input_dict])[feature_columns] #enforce correct column order
  input_scaled = scaler.transform(input_df)

  prediction = model.predict(input_scaled)[0]
  cover_type_names ={
      1: 'Spruce/Fir',
      2: 'Lodgepole Pine',
      3: 'Ponderosa Pine',
      4: 'Cottonwood/Willow',
      5: 'Aspen',
      6: 'Douglas-fir',
      7: 'Krummholz'
  }

  st.success(f"Predicted Cover Type: **{prediction} - {cover_type_names[prediction]}**")
