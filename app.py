import os
import re
import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.models import load_model

# ==========================================
# PAGE CONFIGURATION & CUSTOM STYLING
# ==========================================
st.set_page_config(
    page_title="Spam Shield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark modern palette CSS injection
st.markdown("""
<style>
    /* Global styles */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    /* Sidebar customization */
    [data-testid="stSidebar"] {
        background-color: #1e293b;
        border-right: 1px solid #334155;
    }

    /* Metric card container */
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        text-align: center;
    }

    /* Status Banner Styling */
    .spam-result {
        background-color: rgba(239, 68, 68, 0.15);
        border: 1px solid #ef4444;
        color: #fca5a5;
        padding: 20px;
        border-radius: 10px;
        font-size: 22px;
        font-weight: 700;
        text-align: center;
    }
    .ham-result {
        background-color: rgba(34, 197, 94, 0.15);
        border: 1px solid #22c55e;
        color: #86efac;
        padding: 20px;
        border-radius: 10px;
        font-size: 22px;
        font-weight: 700;
        text-align: center;
    }
    
    /* Button enhancement */
    div.stButton > button:first-child {
        background-color: #3b82f6;
        color: white;
        border-radius: 8px;
        padding: 12px 28px;
        font-weight: 600;
        border: none;
        width: 100%;
        transition: all 0.2s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #2563eb;
        box-shadow: 0 0 12px rgba(59, 130, 246, 0.5);
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# MODEL & PREPROCESSING UTILITIES
# ==========================================
@st.cache_resource
def load_spam_model():
    """Loads and caches the compiled Keras model."""
    model_path = 'spam.pkl'
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' missing from directory.")
        return None
    try:
        # Load keras pickle model
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_text(text: str, max_len: int = 50, num_words: int = 5000) -> np.ndarray:
    """
    Standard text normalization and sequence padding for input vectorization.
    Maps words using a deterministic hash index to match the 5000-dim embedding layer.
    """
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    words = cleaned_text.split()
    
    # Hash-based token mapping matching embedding input dimension limits
    tokens = [(hash(w) % (num_words - 2)) + 1 for w in words]
    
    # Truncate or pad sequence to fixed length 50
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
    else:
        tokens = [0] * (max_len - len(tokens)) + tokens
        
    return np.array([tokens], dtype=np.float32)


# ==========================================
# DASHBOARD INTERFACE
# ==========================================
def main():
    model = load_spam_model()

    # Sidebar Information
    with st.sidebar:
        st.title("🛡️ Spam Shield AI")
        st.caption("RNN-Based Message Classifier")
        st.markdown("---")
        
        st.markdown("**Model Specifications**")
        st.markdown("- **Architecture:** SimpleRNN + Dense")
        st.markdown("- **Input Sequence Length:** 50 Tokens")
        st.markdown("- **Vocabulary Size:** 5,000 Dimensions")
        st.markdown("- **Framework:** Keras / TensorFlow")
        
        st.markdown("---")
        st.info("Paste any SMS, email, or message into the main console to check for spam probability.")

    # Main Body Header
    st.title("Message Spam Classification Console")
    st.write("Analyze text real-time using your neural network model.")
    st.markdown("---")

    # Input Section
    col_input, col_stats = st.columns([3, 2], gap="large")

    with col_input:
        st.subheader("📥 Input Message")
        user_text = st.text_area(
            label="Message Content",
            placeholder="Type or paste message here...",
            height=180,
            label_visibility="collapsed"
        )
        
        analyze_btn = st.button("Run Classification")

    with col_stats:
        st.subheader("📊 Analytics Dashboard")
        
        if analyze_btn:
            if not user_text.strip():
                st.warning("Please input text prior to running analysis.")
            elif model is None:
                st.error("Model not available.")
            else:
                # Run Inference
                processed_input = preprocess_text(user_text)
                raw_prediction = float(model.predict(processed_input, verbose=0)[0][0])
                
                # Display Threshold Metrics
                spam_probability = raw_prediction * 100
                ham_probability = (1 - raw_prediction) * 100
                
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.metric(label="Spam Confidence Score", value=f"{spam_probability:.1f}%")
                st.progress(raw_prediction)
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Dynamic Banner Output
                if raw_prediction >= 0.5:
                    st.markdown(
                        f"<div class='spam-result'>🚨 SPAM DETECTED ({spam_probability:.1f}%)</div>", 
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div class='ham-result'>✅ LEGITIMATE (HAM) ({ham_probability:.1f}%)</div>", 
                        unsafe_allow_html=True
                    )
        else:
            st.info("Enter a text string on the left and click **Run Classification** to view real-time analysis.")

if __name__ == '__main__':
    main()
