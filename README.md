# 🌲 Forest Cover Type Prediction (PRCP-1005)

Multiclass classification project predicting forest cover type from cartographic and environmental variables, using the Roosevelt National Forest (UCI Covertype) dataset.

## 🔗 Live App
👉 [Try the deployed app here](https://forest-cover-prediction-facjh6hohmnwdh7wut54wn.streamlit.app/)

## 📊 Project Overview
- **Task:** Multiclass classification (7 forest cover types)
- **Dataset:** 15,120 rows × 54 features, balanced across 7 classes
- **Approach:** Full EDA → preprocessing → comparison of 6 classification models → hyperparameter tuning with GridSearchCV
- **Best Model:** Random Forest (Tuned)
- **Test Accuracy:** 85.63%

## 🛠️ Tech Stack
- **Language:** Python
- **ML/Data:** scikit-learn, pandas, numpy, joblib
- **Deployment:** Streamlit Community Cloud
- **Model Persistence:** joblib (compressed)

## 🚀 App Features
Interactive form to input:
- Terrain features: Elevation, Aspect, Slope
- Distance features: Horizontal/Vertical Distance to Hydrology, Roadways, Fire Points
- Hillshade indices (9am, Noon, 3pm)
- Wilderness Area and Soil Type (categorical selection)

Returns predicted **Cover_Type** (1–7) based on the trained Random Forest model.

## 💻 Run Locally

```bash
git clone https://github.com/NehaYadav1997/forest-cover-prediction.git
cd forest-cover-prediction
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Repository Structure
