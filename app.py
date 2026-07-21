import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Face Mask Detection", page_icon="😷")

st.title("😷 Face Mask Detection")

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_my_model():
      return load_model("mask_final.keras")

model = load_my_model()

# -----------------------------
# Session State
# -----------------------------
if "open_camera" not in st.session_state:
      st.session_state.open_camera = False

# -----------------------------
# Prediction Function
# -----------------------------
def predict_mask(img):

# Convert camera image to RGB
      img = img.convert("RGB")

# Resize
      img = img.resize((128, 128))

# Show processed image
      st.image(img, caption="Image Used for Prediction", width=250)

# Convert to array
      img_array = image.img_to_array(img)

#Normalize
      img_array = img_array / 255.0

# Add batch dimension
      img_array = np.expand_dims(img_array, axis=0)

# Predict
      prediction = model.predict(img_array, verbose=0)

      prob = float(prediction[0][0])

# Show prediction value
      st.write("### Raw Prediction Value")
      st.write(prob)

# Prediction
      if prob > 0.5:
            st.error(f"❌ WITHOUT MASK ({prob:.2%})")
      else:
            st.success(f"✅ WITH MASK ({(1-prob):.2%})")


# ===============================
# Upload Image
# ===============================
st.header("📁 Upload Image")

uploaded_file = st.file_uploader(
      "Choose an image",
      type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

      img = Image.open(uploaded_file)

      st.image(img, caption="Uploaded Image", use_container_width=True)

      predict_mask(img)

st.markdown("---")

# ===============================
# Camera Buttons
# ===============================
col1, col2 = st.columns(2)

with col1:
      if st.button("📸 Open Camera"):
            st.session_state.open_camera = True

with col2:
      if st.button("❌ Close Camera"):
            st.session_state.open_camera = False

# ===============================
# Camera
# ===============================
if st.session_state.open_camera:

      camera_image = st.camera_input("Take a Photo")

      if camera_image is not None:

            img = Image.open(camera_image)

            st.image(img, caption="Captured Image", use_container_width=True)

            predict_mask(img)