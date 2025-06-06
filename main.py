import gradio as gr
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import base64

# Configure environment for better compatibility
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

try:
    import tensorflow as tf
    from tensorflow import keras
    import tensorflow.keras.backend as K
    
    # Configure TensorFlow
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    
    # Disable GPU to avoid memory issues on HF Spaces
    tf.config.set_visible_devices([], 'GPU')
    
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Define the custom loss function
def focal_loss_fixed(y_true, y_pred):
    gamma = 2.0
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

# Global model variable
model = None

def load_model():
    """Load the melanoma detection model"""
    global model
    if not TF_AVAILABLE:
        return "TensorFlow not available. Please install tensorflow."
    
    try:
        if model is None:
            # Clear any existing sessions
            tf.keras.backend.clear_session()
            
            # Load model
            model = keras.models.load_model(
                'skin_cancer_detector.keras', 
                custom_objects={'focal_loss_fixed': focal_loss_fixed},
                compile=False
            )
            
            # Recompile with standard settings
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        
        return "Model loaded successfully!"
    except Exception as e:
        return f"Error loading model: {str(e)}"

def preprocess_image(image):
    """Preprocess image for model prediction"""
    if image is None:
        return None
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Resize to model input shape
    img_resized = cv2.resize(img_array, (224, 224))
    
    # Normalize pixel values
    img_normalized = img_resized / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch

def create_result_plot(predictions, class_names):
    """Create a horizontal bar plot showing prediction confidence"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#2E8B57', '#FF8C00', '#DC143C']  # Green, Orange, Red
    
    # Create horizontal bar chart
    bars = ax.barh(class_names, predictions, color=colors)
    
    # Add percentage labels on bars
    for bar, pred in zip(bars, predictions):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{pred:.1%}', ha='left', va='center', fontweight='bold')
    
    ax.set_xlabel('Confidence Level', fontsize=12, fontweight='bold')
    ax.set_title('Melanoma Classification Results', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    
    # Format x-axis as percentage
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    
    # Add grid for better readability
    ax.grid(axis='x', alpha=0.3)
    
    # Style the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig

def predict_melanoma(image):
    """Main prediction function"""
    if image is None:
        return "Please upload an image.", None, None
    
    if not TF_AVAILABLE or model is None:
        load_status = load_model()
        if "Error" in load_status:
            return load_status, None, None
    
    try:
        # Preprocess the image
        processed_image = preprocess_image(image)
        if processed_image is None:
            return "Error processing image.", None, None
        
        # Get predictions
        with tf.device('/CPU:0'):  # Force CPU usage
            predictions = model.predict(processed_image, verbose=0)
        
        confidence_levels = predictions[0]
        class_names = ['Nevus\n(Benign)', 'Atypical\n(Dysplastic)', 'Melanoma\n(Malignant)']
        
        # Get the predicted class
        predicted_index = np.argmax(confidence_levels)
        predicted_class = ['Nevus', 'Atypical', 'Melanoma'][predicted_index]
        confidence = confidence_levels[predicted_index]
        
        # Create result message
        class_descriptions = {
            0: "🟢 **NEVUS (Benign)**\n\nThis appears to be a common mole (nevus). While generally harmless, continue regular skin monitoring.",
            1: "🟡 **ATYPICAL (Dysplastic)**\n\nThis lesion shows atypical features. Professional evaluation recommended.",
            2: "🔴 **MELANOMA (Malignant)**\n\n⚠️ **URGENT**: This lesion shows features concerning for melanoma. Seek immediate dermatological evaluation."
        }
        
        result_text = f"""
## Prediction Results

{class_descriptions[predicted_index]}

**Confidence: {confidence:.1%}**

---

### Detailed Analysis:
- **Nevus (Benign)**: {confidence_levels[0]:.1%}
- **Atypical (Dysplastic)**: {confidence_levels[1]:.1%}  
- **Melanoma (Malignant)**: {confidence_levels[2]:.1%}

---

⚠️ **Medical Disclaimer**: This AI tool is for educational purposes only and should not replace professional medical diagnosis. Always consult qualified healthcare providers for proper medical evaluation.
        """
        
        # Create visualization
        plot = create_result_plot(confidence_levels, class_names)
        
        return result_text, plot, f"Predicted: {predicted_class} ({confidence:.1%} confidence)"
        
    except Exception as e:
        return f"Error during prediction: {str(e)}", None, None

# Custom CSS for better styling
custom_css = """
#component-0 {
    max-width: 900px;
    margin: auto;
}

.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.output-markdown h2 {
    color: #2c3e50;
    border-bottom: 2px solid #3498db;
    padding-bottom: 10px;
}

.output-markdown h3 {
    color: #34495e;
}

footer {
    visibility: hidden;
}
"""

# Example images for demonstration (you can add these to your HF repo)
examples = [
    # Add paths to example images here
    # ["example1.jpg"],
    # ["example2.jpg"],
    # ["example3.jpg"],
]

# Create the Gradio interface
def create_interface():
    # Load model on startup
    load_model()
    
    with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as interface:
        
        gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1 style="color: #2c3e50; margin-bottom: 10px;">🔬 Melanoma Detection System</h1>
            <p style="color: #7f8c8d; font-size: 18px;">AI-Powered Skin Lesion Classification</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                           color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                    <h3 style="margin: 0; color: white;">ℹ️ About This Tool</h3>
                    <p style="margin: 10px 0; color: white;">This AI model classifies skin lesions into:</p>
                    <ul style="color: white; margin: 10px 0;">
                        <li><strong>Nevus (Benign)</strong> - Common moles, generally harmless</li>
                        <li><strong>Atypical (Dysplastic)</strong> - Unusual moles requiring monitoring</li>
                        <li><strong>Melanoma (Malignant)</strong> - Potentially cancerous lesions</li>
                    </ul>
                </div>
                """)
                
                image_input = gr.Image(
                    type="pil",
                    label="📷 Upload Skin Lesion Image",
                    height=300
                )
                
                predict_button = gr.Button(
                    "🔍 Analyze Image", 
                    variant="primary",
                    size="lg"
                )
                
                gr.HTML("""
                <div style="background: #fff3cd; border: 1px solid #ffeaa7; 
                           border-radius: 5px; padding: 15px; margin-top: 20px;">
                    <strong>⚠️ Important:</strong> This tool is for educational purposes only. 
                    Always consult a qualified dermatologist for professional medical diagnosis.
                </div>
                """)
            
            with gr.Column(scale=1):
                result_output = gr.Markdown(
                    label="📊 Analysis Results",
                    value="Upload an image and click 'Analyze' to see results."
                )
                
                plot_output = gr.Plot(
                    label="📈 Confidence Visualization"
                )
                
                status_output = gr.Textbox(
                    label="Status",
                    interactive=False
                )
        
        # Add examples if available
        if examples:
            gr.Examples(
                examples=examples,
                inputs=image_input,
                label="📋 Example Images"
            )
        
        # Connect the predict function
        predict_button.click(
            fn=predict_melanoma,
            inputs=image_input,
            outputs=[result_output, plot_output, status_output]
        )
        
        gr.HTML("""
        <div style="text-align: center; margin-top: 30px; padding: 20px; 
                   background: #f8f9fa; border-radius: 10px;">
            <h3>🏥 Medical Disclaimer</h3>
            <p>This AI tool is designed for educational and research purposes only. 
            It should not be used as a substitute for professional medical advice, 
            diagnosis, or treatment. Always seek the advice of qualified healthcare 
            providers with questions about medical conditions.</p>
        </div>
        """)
    
    return interface

# Create and launch the interface
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True)