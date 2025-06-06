import streamlit as st
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import cv2
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Set page config
st.set_page_config(
    page_title="Melanoma Detection System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    
    .result-card {
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    .benign-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .malignant-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    
    .atypical-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
    
    .confidence-text {
        font-size: 1.8rem;
        font-weight: bold;
        text-align: center;
    }
    
    .class-text {
        font-size: 1.4rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Define the custom loss function
def focal_loss_fixed(y_true, y_pred):
    gamma = 2.0
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

# Load model function with caching
@st.cache_resource
def load_model():
    try:
        model = keras.models.load_model('skin_cancer_detector.keras', 
                                      custom_objects={'focal_loss_fixed': focal_loss_fixed})
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("Please make sure 'skin_cancer_detector.keras' is in the same directory as this script.")
        return None

# Preprocess image function
def preprocess_image(image):
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Resize to model input shape
    img_resized = cv2.resize(img_array, (224, 224))
    
    # Normalize pixel values
    img_normalized = img_resized / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch

# Create confidence chart
def create_confidence_chart(predictions, class_names):
    colors = ['#667eea', '#4facfe', '#f5576c']
    
    fig = go.Figure(data=[
        go.Bar(x=class_names, y=predictions, 
               marker_color=colors,
               text=[f'{p:.2%}' for p in predictions],
               textposition='auto')
    ])
    
    fig.update_layout(
        title="Prediction Confidence Levels",
        xaxis_title="Classification",
        yaxis_title="Confidence",
        yaxis=dict(tickformat='.0%'),
        height=400,
        showlegend=False
    )
    
    return fig

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🔬 Melanoma Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Skin Lesion Classification</p>', unsafe_allow_html=True)
    
    # Sidebar information
    with st.sidebar:
        st.header("ℹ️ About This Tool")
        st.markdown("""
        This AI model classifies skin lesions into three categories:
        
        **🟢 Nevus (Benign)**
        - Common moles
        - Generally harmless
        
        **🟡 Atypical (Dysplastic)**
        - Unusual moles
        - May require monitoring
        
        **🔴 Melanoma (Malignant)**
        - Potentially cancerous
        - Requires immediate medical attention
        """)
        
        st.markdown("---")
        st.markdown("**⚠️ Important Disclaimer**")
        st.markdown("""
        This tool is for educational purposes only. 
        Always consult a qualified dermatologist for 
        professional medical diagnosis.
        """)
    
    # Load model
    model = load_model()
    if model is None:
        st.stop()
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📷 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a skin lesion image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of the skin lesion for analysis"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Add analyze button
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Analyzing image... Please wait."):
                    # Preprocess image
                    processed_image = preprocess_image(image)
                    
                    # Get predictions
                    predictions = model.predict(processed_image, verbose=0)
                    confidence_levels = predictions[0]
                    
                    # Store results in session state
                    st.session_state['predictions'] = confidence_levels
                    st.session_state['analyzed'] = True
    
    with col2:
        st.header("📊 Analysis Results")
        
        if 'analyzed' in st.session_state and st.session_state['analyzed']:
            predictions = st.session_state['predictions']
            class_names = ['Nevus', 'Atypical', 'Melanoma']
            class_descriptions = ['Benign (Safe)', 'Dysplastic (Monitor)', 'Malignant (Urgent)']
            
            # Get predicted class
            predicted_index = np.argmax(predictions)
            predicted_class = class_names[predicted_index]
            confidence = predictions[predicted_index]
            
            # Display main prediction
            if predicted_index == 0:  # Nevus
                st.markdown(f"""
                <div class="result-card benign-card">
                    <div class="class-text">🟢 {predicted_class}</div>
                    <div class="class-text">{class_descriptions[predicted_index]}</div>
                    <div class="confidence-text">{confidence:.1%} Confidence</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="info-box">
                    <strong>Assessment: Likely Benign</strong><br>
                    This appears to be a common mole (nevus). While generally harmless, 
                    continue regular skin monitoring and consult a dermatologist if you notice any changes.
                </div>
                """, unsafe_allow_html=True)
                
            elif predicted_index == 1:  # Atypical
                st.markdown(f"""
                <div class="result-card atypical-card">
                    <div class="class-text">🟡 {predicted_class}</div>
                    <div class="class-text">{class_descriptions[predicted_index]}</div>
                    <div class="confidence-text">{confidence:.1%} Confidence</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="warning-box">
                    <strong>Assessment: Atypical/Dysplastic</strong><br>
                    This lesion shows atypical features. While not necessarily cancerous, 
                    it requires professional evaluation and may need regular monitoring.
                </div>
                """, unsafe_allow_html=True)
                
            else:  # Melanoma
                st.markdown(f"""
                <div class="result-card malignant-card">
                    <div class="class-text">🔴 {predicted_class}</div>
                    <div class="class-text">{class_descriptions[predicted_index]}</div>
                    <div class="confidence-text">{confidence:.1%} Confidence</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="warning-box">
                    <strong>⚠️ Assessment: Suspicious for Melanoma</strong><br>
                    This lesion shows features concerning for melanoma. 
                    <strong>Seek immediate dermatological evaluation.</strong>
                </div>
                """, unsafe_allow_html=True)
            
            # Display confidence chart
            st.plotly_chart(
                create_confidence_chart(predictions, class_names), 
                use_container_width=True
            )
            
            # Detailed breakdown
            st.subheader("📋 Detailed Analysis")
            for i, (class_name, desc, conf) in enumerate(zip(class_names, class_descriptions, predictions)):
                if i == 0:
                    emoji = "🟢"
                elif i == 1:
                    emoji = "🟡"
                else:
                    emoji = "🔴"
                    
                st.write(f"{emoji} **{class_name}** ({desc}): {conf:.2%}")
        
        else:
            st.info("👆 Upload an image and click 'Analyze' to see results")
    
    # Footer with medical disclaimer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        <strong>⚠️ Medical Disclaimer:</strong> This AI tool is for educational and screening purposes only. 
        It should not replace professional medical diagnosis. Always consult qualified healthcare providers 
        for proper medical evaluation and treatment decisions.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()