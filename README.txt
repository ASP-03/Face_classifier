🎭 FairFace Streamlit App

This is a simple and intuitive Streamlit app that classifies race (Asian, Black, or White) and gender (Male or Female) using deep learning models trained on the FairFace dataset.

──────────────────────────────
📁 Files

Everything should be in ONE folder:

- app.py                         ← Main Streamlit app
- fairface_race_classifier.h5   ← Trained Keras model for race prediction
- fairface_gender_classifier.h5 ← Trained Keras model for gender prediction

──────────────────────────────
🚀 How to Run

1. Place all required files in the same folder:
   - app.py
   - fairface_race_classifier.h5
   - fairface_gender_classifier.h5

2. Install the dependencies:
   pip install streamlit tensorflow pillow

3. Run the app using Streamlit:
   streamlit run app.py

4. Once the app is running:
   - Upload a face image
   - Get instant predictions for:
     • Race: Asian, Black, or White
     • Gender: Male or Female

──────────────────────────────
✨ Features

- Upload your own face image
- Predict race and gender instantly
- Clean UI with real-time prediction
- Powered by your trained deep learning models

──────────────────────────────
🧠 Built With

- Streamlit – for the web app interface
- TensorFlow / Keras – for model handling
- FairFace Dataset – for training data

──────────────────────────────
📌 Note

This app is intended for educational and demonstration purposes only.
Prediction accuracy may vary depending on training quality and data balance.

──────────────────────────────
📄 License

This project is open-source and available under the MIT License.
