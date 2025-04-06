import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image
import os
import time


if 'sample_selected' not in st.session_state:
    st.session_state.sample_selected = None


try:
    race_model = load_model("fairface_race_classifier.h5")
    gender_model = load_model("fairface_gender_classifier.h5")
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()


race_labels = ['Asian', 'Black', 'White']
gender_labels = ['Female', 'Male']


st.set_page_config(
    page_title="FairFace Classifier",
    layout="centered",
    page_icon="👥"
)

st.title("👥 Face Attribute Predictor")
st.subheader("Race & Gender Classification")

def reset_all():
    st.session_state.sample_selected = None
 
    st.rerun()


uploaded_file = st.file_uploader(
    "Choose a face image (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"],
    help="Clear frontal face images work best",
    key="file_uploader"  
)


if st.session_state.sample_selected:
    try:
        uploaded_file = open(st.session_state.sample_selected, 'rb')
    except:
        st.error("Failed to load sample image")
        st.session_state.sample_selected = None


if uploaded_file is not None:
    try:
        with st.spinner('Processing image...'):
            
            img_display = Image.open(uploaded_file)
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(img_display, caption="Uploaded Image", use_container_width=True)
                
                if st.button("🔄 Reset Image", key="reset_button"):
                    reset_all()
                    st.rerun()
           
            img = Image.open(uploaded_file).convert('RGB').resize((224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            
            if img_array.shape != (1, 224, 224, 3):
                st.error(f"Unexpected image shape: {img_array.shape}")
                st.stop()
            
           
            race_pred = race_model.predict(img_array, verbose=0)
            gender_pred = gender_model.predict(img_array, verbose=0)[0][0]
            
           
            race_idx = np.argmax(race_pred)
            race_label = race_labels[race_idx]
            race_conf = np.max(race_pred) * 100
            
            gender_label = gender_labels[int(gender_pred > 0.5)]
            gender_conf = gender_pred if gender_pred > 0.5 else 1 - gender_pred
            gender_conf *= 100
            
            
            with col2:
                st.subheader("Prediction Results")
                st.metric("Race", race_label, f"{race_conf:.1f}% confidence")
                st.metric("Gender", gender_label, f"{gender_conf:.1f}% confidence")
                st.progress(int(race_conf)/100, text="Race Confidence")
                st.progress(int(gender_conf)/100, text="Gender Confidence")
    
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
    finally:
        if uploaded_file:
            uploaded_file.close()

if uploaded_file is None:
    st.info("ℹ️ Try these sample images:")
    sample_images = [f for f in ["sample1.jpg", "sample2.jpg", "sample3.jpg"] if os.path.exists(f)]
    
    if sample_images:
        cols = st.columns(len(sample_images))
        for col, sample in zip(cols, sample_images):
            with col:
                st.image(sample, use_container_width=True)
                if st.button(f"Use {sample}"):
                    st.session_state.sample_selected = sample
                    st.rerun()
    else:
        st.warning("No sample images found")

#
st.markdown(
    """
    <style>
        body { 
            background-color: #F6E6CB; 
            font-family: 'Arial', sans-serif;
        }
        
        .stTextInput, .stFileUploader { 
            border: 2px solid rgba(255, 255, 255, 0.2); 
            border-radius: 10px; 
            backdrop-filter: blur(10px);
            background: rgba(255, 255, 255, 0.1);
            padding: 10px;
        }

        .prediction-box { 
            padding: 20px; 
            border-radius: 15px; 
            margin-top: 20px; 
            text-align: center;
            background: rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(12px);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        .sidebar .sidebar-content { 
            background: rgba(255, 255, 255, 0.1); 
            padding: 15px; 
            border-radius: 10px;
            backdrop-filter: blur(10px);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        .stButton>button { 
            background: linear-gradient(135deg, rgba(255,255,255,0.2), rgba(255,255,255,0.1));
            border: none;
            color: white; 
            border-radius: 10px; 
            padding: 10px;
            backdrop-filter: blur(10px);
            transition: all 0.3s ease-in-out;
        }

        .stButton>button:hover {
            background: rgba(255, 255, 255, 0.3);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.15);
        }
    </style>
    """,
    unsafe_allow_html=True
)


with st.expander("ℹ️ About this app & limitations"):
    st.markdown("""
    **App Features:**
    
    - 🎯 **Race Prediction**: Classifies faces as Asian, Black, or White
    - ♀️♂️ **Gender Prediction**: Identifies as Male or Female
    - 📊 **Confidence Metrics**: Shows prediction confidence levels
    - 🖼️ **Image Guidelines**: Works best with clear frontal face images
    
    **Limitations:**
    
    - 📉 **Accuracy Variance**: Results may vary with image quality and angle
    - 🏷️ **Categories**: Limited to specified race/gender categories
    - ⚠️ **Purpose**: For demonstration purposes only
    
    **Privacy Notice:**
    
    - 🔒 **No Storage**: Images are processed in memory and not stored
    - 🌐 **No Uploads**: All processing happens locally in your browser
    """)

