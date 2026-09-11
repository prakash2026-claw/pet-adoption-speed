import os
import tempfile
import pandas as pd
import streamlit as st
from PIL import Image
from autogluon.multimodal import MultiModalPredictor

st.set_page_config(page_title="MultiModal AI App", layout="centered")
st.title("🖼️ AutoGluon Vision + NLP Predictor")

# 1. Cache the MultiModal Model
@st.cache_resource
def load_multimodal_model():
    return MultiModalPredictor.load("AutogluonModels/ag-multimodal-model/")

try:
    predictor = load_multimodal_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# 2. Define Preset Examples
EXAMPLES = {
    "Example 1: Cute Cat": {
        "image_path": "sample_cat.jpg",
        "text": "A fluffy orange cat sleeping on a sunny windowsill."
    },
    "Example 2: Sports Car": {
        "image_path": "sample_car.jpg",
        "text": "A sleek red sports car driving down an empty highway."
    }
}

# 3. Interactive Input Selector
st.subheader("Step 1: Choose Your Input Method")
input_mode = st.radio(
    "Select how you want to provide data:",
    ["Use a Pre-loaded Example", "Upload Custom Image & Text"],
    horizontal=True
)

# Initialize variables to hold final values for the model
final_image = None
final_text = ""
is_example_mode = (input_mode == "Use a Pre-loaded Example")

# 4. Handle UI layout based on selection
if is_example_mode:
    selected_example = st.selectbox("Choose an example scenario:", list(EXAMPLES.keys()))
    
    # Load preset data
    example_data = EXAMPLES[selected_example]
    final_text = example_data["text"]
    
    # Attempt to open local image file
    if os.path.exists(example_data["image_path"]):
        final_image = Image.open(example_data["image_path"])
        st.image(final_image, caption=f"Selected: {selected_example}", width=350)
        st.info(f"**Preset Text:** {final_text}")
    else:
        st.error(f"Missing sample file: {example_data['image_path']}. Please place it in the app directory.")

else:
    # Custom User Upload Mode
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### **Column 1: Image**")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            final_image = Image.open(uploaded_file)
            st.image(final_image, caption="Uploaded Image Preview", use_container_width=True)
            
    with col2:
        st.markdown("### **Column 2: Text**")
        final_text = st.text_area("Enter associated description/text:", placeholder="Type context here...")

# 5. Prediction Logic
st.markdown("---")
if st.button("🚀 Run Multimodal Prediction", type="primary"):
    if final_image is None:
        st.warning("Please ensure an image is loaded or uploaded.")
    elif not final_text.strip():
        st.warning("Please ensure text context is provided.")
    else:
        with st.spinner("Processing Multimodal fusion model..."):
            # Create a safe temporary path for the image file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                # Save the PIL image object to the temp file
                final_image.convert("RGB").save(temp_file.name, format="JPEG")
                temp_image_path = temp_file.name

            try:
                # Construct data matching your exact training column headers
                input_data = pd.DataFrame([{
                    'image_column_name': temp_image_path,
                    'text_column_name': final_text
                }])

                # Run inference
                prediction = predictor.predict(input_data)
                
                # Display Results
                st.subheader("🎯 Prediction Output")
                st.success(f"**Result:** {prediction.iloc[0]}")
                
            except Exception as predict_error:
                st.error(f"Prediction failed: {predict_error}")
                
            finally:
                # Safe cleanup of the temporary file
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
