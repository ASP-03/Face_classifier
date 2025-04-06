# FairFace Streamlit App

This is a simple Streamlit app to classify race (Asian, Black, or White) and gender (Male, Female) using a deep learning model trained on the FairFace dataset.

## 📦 Files
- `app.py`: Main Streamlit app
- `fairface_race_classifier.h5`: Your trained Keras model for race prediction
- `fairface_race_classifier.h5`: Your trained Keras model for gender prediction

## 🚀 How to Run(all files need to be in the same folder)

1. Move your trained model file (`fairface_race_classifier.h5`) into this folder.
2. Move your trained model file (`fairface_gender_classifier.h5`) into this folder.
3. Install dependencies:
   ```
   pip install streamlit tensorflow pillow
   ```
4. Run the app:
   ```
   streamlit run app.py
Enjoy!
