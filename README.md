# 🌾 Agri-Tech ML Hub: Smart Crop and Fertilizer Recommendation System

The **Agri-Tech ML Hub** is a two-part application designed to assist farmers and agricultural enthusiasts in making data-driven decisions. It provides:

- **Crop Recommendation**: Suggests the most suitable crop to grow based on soil nutrients (N, P, K), environmental factors (temperature, humidity, pH, rainfall), and soil type.  
- **Fertilizer Recommendation**: Provides the ideal NPK (Nitrogen, Phosphorus, Potassium) ratio required for a chosen crop.

The application is structured using a **Streamlit frontend** for the user interface and a separate **Flask backend** to handle the machine learning predictions and data lookups.

---

## 🌟 Features

- **Crop Suitability Prediction**: Uses a pre-trained Random Forest model (`crop_model.pkl`).  
- **Fertilizer Ratio Lookup**: Provides optimal NPK ratios from a static table (`fertilizer_ratios.pkl`).  
- **User Authentication**: Basic sign-up and login via `streamlit-authenticator`.  
- **Multilingual Support**: Easy language translation in the Streamlit app.  
- **Responsive UI**: Custom CSS for a visually appealing wide-screen layout.

---

## ⚙️ System Architecture

### Streamlit Frontend (`app.py`)
- Handles user interaction, input forms, and results display.  
- Manages user authentication and session state.  
- Sends API requests to Flask backend for ML predictions and lookups.

### Flask Backend (`flask_backend.py`)
- Lightweight REST API server.  
- Loads ML models (`crop_model.pkl`, `crop_encoder.pkl`) and fertilizer ratios.  
- Processes incoming data, performs feature engineering, and returns predictions.

---

## 🚀 Setup and Installation

1. **Prerequisites**
- Python 3.8+ installed.

2. **Clone the Repository**
- git clone https://github.com/AnanyaB-262005/Crop_Reccomendation_Using_ML
- cd agri-tech-ml-hub


3. **Install Dependencies**
- pip install -r requirements.txt

4.**Run the Flask Backend**
- python flask_backend.py

5.**Run the Streamlit Frontend**
- streamlit run app.py

## 🖥️ Usage and Screenshots 
1.**Login / Signup**
- Log in or create a new account via the sidebar menu.
### Login Page
![Login Page Screenshot](screenshots/login_page.png)

2.**Crop Recommendation**
- Select "Crop Recommendation".
- Enter soil nutrients (N, P, K), temperature, humidity, pH, rainfall, and soil type.
- Click "Predict Crop".
### Crop Recommendation
![Crop Recommendation Screenshot](screenshots/crop_recommendation.png)


3.**Fertilizer Recommendation**
- Select "Fertilizer Recommendation".
- Select a crop.
- Click "Get NPK Ratio".
### Fertilizer Recommendation
![Fertilizer Recommendation Screenshot](screenshots/fertilizer_recommendation.png)


## Project Structure

| File                                | Description                              |
| ----------------------------------- | ---------------------------------------- |
| `app.py`                            | Streamlit frontend (UI, Auth, API calls) |
| `flask_backend.py`                  | Flask API backend                        |
| `requirements.txt`                  | Python dependencies                      |
| `crop_model.pkl`                    | Pre-trained Random Forest model          |
| `crop_encoder.pkl`                  | Label encoder for crops                  |
| `fertilizer_ratios.pkl`             | NPK ratios lookup                        |
| `Crop_recommendation_with_soil.csv` | Original dataset                         |
| `model_features.pkl`                | Model input features                     |
| `img2.jpg`                          | Background image                         |
| `screenshots/`                      | Screenshots folder                       |

## 🤝 Contribution

Feel free to open issues or submit pull requests to improve the model accuracy, add more language translations, or enhance the user interface!

