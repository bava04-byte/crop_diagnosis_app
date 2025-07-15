import streamlit as st

# ✅ Must be first Streamlit call
st.set_page_config(page_title="Smart Crop Issue Detector", layout="centered")

# 📦 Other imports
from PIL import Image, UnidentifiedImageError
from utils.diagnose import analyze_crop_issue
from utils.metrics import get_pytorch_model_accuracy
from googletrans import Translator

# --- Title & Instructions ---
st.title("🌾 Smart Crop Issue Detector")
st.markdown("Upload an image of your crop and describe the problem using text.")

# --- Sidebar: Model Accuracy ---
with st.sidebar:
    st.markdown("## 📊 Model Accuracy")
    try:
        acc = get_pytorch_model_accuracy()
        st.success(f"🖼️ Image Model Accuracy: {acc}%")
    except Exception as e:
        st.error(f"⚠️ Accuracy check failed: {e}")

# --- Crop Type ---
crop_type = st.selectbox("🌱 Select Crop Type", ["Tomato", "Potato"])

# --- Output Language ---
language_map = {
    "English": "en",
    "Malayalam": "ml",
    "Hindi": "hi",
    "Tamil": "ta",
    "Telugu": "te",
    "Kannada": "kn"
}
selected_lang = st.selectbox("🌐 Choose Output Language", list(language_map.keys()))
lang_code = language_map[selected_lang]

# --- Image Upload ---
with st.expander("🖼️ Upload Crop Image"):
    uploaded_file = st.file_uploader("Upload Crop Image", type=["jpg", "jpeg", "png"])

# --- Text Description ---
st.subheader("✍️ Describe the Problem")
description = st.text_area(
    "Type crop symptoms (e.g., yellow leaves, black spots, white powder, curling, dryness, insects)..."
)

# --- Translator Function ---
def translate_text(text, lang_code):
    try:
        translator = Translator()
        return translator.translate(text, dest=lang_code).text
    except Exception as e:
        return f"⚠️ Translation failed: {e}"

# --- Analyze Button ---
st.header("🔬 Get Diagnosis")

if st.button("Analyze Now"):
    # ✅ Step 1: Check upload
    if not uploaded_file:
        st.warning("⚠️ Please upload a crop image first.")
        st.stop()

    # ✅ Step 2: Check file type and open safely
    try:
        # Rewind file in case Streamlit messed with it
        uploaded_file.seek(0)
        image = Image.open(uploaded_file).convert("RGB")
    except UnidentifiedImageError:
        st.error("❌ The uploaded file is not a valid image. Please upload a valid JPG or PNG.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Could not open the uploaded image: {e}")
        st.stop()

    # ✅ Step 3: Check text input
    final_input = description.strip()
    if not final_input:
        st.warning("⚠️ Please describe the issue using text.")
        st.stop()

    # ✅ Step 4: Analyze safely
    with st.spinner("🔬 Analyzing the crop issue..."):
        try:
            diagnosis, solution = analyze_crop_issue(image, final_input, crop_type)
        except Exception as e:
            st.error(f"❌ Analysis failed: {e}")
            st.stop()

    # ✅ Step 5: Translate if needed
    if lang_code != "en":
        try:
            diagnosis = translate_text(diagnosis, lang_code)
            solution = translate_text(solution, lang_code)
        except Exception as e:
            st.error(f"⚠️ Translation failed: {e}")

    # ✅ Step 6: Final type guard for display
    if isinstance(image, Image.Image):
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="📷 Uploaded Crop Image", use_container_width=True)

        with col2:
            st.markdown("### 🧪 Diagnosis")
            st.success(diagnosis)

            st.markdown("### 💡 Suggested Action")
            st.info(solution)
    else:
        st.error("❌ Something went wrong — the image object is not valid.")
