import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    st.warning("OpenCV not available. Some features may be limited.")


# Set page config
st.set_page_config(
    page_title="Melanoma Detection AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .benign {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .malignant {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    .atypical {
        background-color: #fff3cd;
        color: #856404;
        border: 2px solid #ffeaa7;
    }
    .confidence-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 20px 0;
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

@st.cache_resource
def load_model():
    """Load the melanoma detection model"""
    try:
        model = keras.models.load_model('skin_cancer_detector.keras', 
                                      custom_objects={'focal_loss_fixed': focal_loss_fixed})
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image):
    """Preprocess the uploaded image for model prediction"""
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Resize to model input shape
    img_resized = cv2.resize(img_array, (224, 224))
    
    # Normalize pixel values
    img_normalized = img_resized / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch

def create_confidence_chart(predictions, class_names):
    """Create a confidence chart using Plotly"""
    colors = ['#28a745', '#ffc107', '#dc3545']  # Green, Yellow, Red
    
    fig = go.Figure(data=[
        go.Bar(
            x=class_names,
            y=predictions,
            marker_color=colors,
            text=[f'{pred:.2%}' for pred in predictions],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Confidence Levels",
        xaxis_title="Class",
        yaxis_title="Confidence",
        yaxis=dict(range=[0, 1]),
        height=400,
        showlegend=False
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🔬 Melanoma Detection AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced AI-powered skin lesion classification</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown("""
        1. **Upload an image** of a skin lesion
        2. **Wait for analysis** (few seconds)
        3. **Review results** and confidence levels
        4. **Consult a dermatologist** for professional diagnosis
        """)
        
        st.header("🏥 Classification Types")
        st.markdown("""
        - **Nevus**: Benign (harmless) mole
        - **Atypical**: Unusual but not necessarily cancerous
        - **Melanoma**: Malignant (cancerous) tumor
        """)
        
        st.markdown("---")
        st.markdown("*This tool is for educational purposes only and should not replace professional medical advice.*")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📁 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a skin lesion image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of the skin lesion for analysis"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Image info
            st.markdown(f"**Image size:** {image.size[0]} x {image.size[1]} pixels")
    
    with col2:
        if uploaded_file is not None:
            st.header("🔍 Analysis Results")
            
            # Load model
            model = load_model()
            
            if model is not None:
                with st.spinner("Analyzing image... Please wait"):
                    try:
                        # Preprocess image
                        processed_image = preprocess_image(image)
                        
                        # Get predictions
                        predictions = model.predict(processed_image, verbose=0)
                        confidence_levels = predictions[0]
                        
                        # Class names
                        class_names = ['Nevus', 'Atypical', 'Melanoma']
                        
                        # Get predicted class
                        predicted_class_index = np.argmax(confidence_levels)
                        predicted_class = class_names[predicted_class_index]
                        predicted_confidence = confidence_levels[predicted_class_index]
                        
                        # Display main result
                        if predicted_class == 'Nevus':
                            st.markdown(f'<div class="result-box benign">✅ BENIGN<br>{predicted_class}</div>', 
                                      unsafe_allow_html=True)
                        elif predicted_class == 'Melanoma':
                            st.markdown(f'<div class="result-box malignant">⚠️ MALIGNANT<br>{predicted_class}</div>', 
                                      unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="result-box atypical">⚡ ATYPICAL<br>{predicted_class}</div>', 
                                      unsafe_allow_html=True)
                        
                        # Confidence level
                        st.markdown(f'''
                        <div class="confidence-box">
                            <strong>Confidence Level: {predicted_confidence:.1%}</strong>
                        </div>
                        ''', unsafe_allow_html=True)
                        
                        # Create and display confidence chart
                        fig = create_confidence_chart(confidence_levels, class_names)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Detailed results
                        with st.expander("📊 Detailed Confidence Scores"):
                            results_df = pd.DataFrame({
                                'Class': class_names,
                                'Confidence': [f'{conf:.4f}' for conf in confidence_levels],
                                'Percentage': [f'{conf:.2%}' for conf in confidence_levels]
                            })
                            st.dataframe(results_df, use_container_width=True)
                        
                        # Medical disclaimer
                        st.markdown('''
                        <div class="warning-box">
                            <strong>⚠️ Medical Disclaimer</strong><br>
                            This AI tool is for educational and screening purposes only. 
                            Always consult with a qualified dermatologist or healthcare 
                            professional for proper diagnosis and treatment.
                        </div>
                        ''', unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
            else:
                st.error("❌ Model could not be loaded. Please check if 'skin_cancer_detector.keras' exists in the current directory.")
        else:
            st.info("👆 Please upload an image to start the analysis")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 0.9rem;'>"
        "🔬 Powered by TensorFlow & Streamlit | "
        "Remember: This is not a substitute for professional medical advice"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()