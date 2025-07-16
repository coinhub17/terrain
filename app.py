
import streamlit as st
from PIL import Image
import io
import os
import base64
import numpy as np
from dotenv import load_dotenv
import streamlit.components.v1 as components
import os
from huggingface_hub import InferenceClient
import streamlit.components.v1 as components
load_dotenv()
# --- Hugging Face Client Initialization ---
HF_TOKEN =os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    st.error("Hugging Face API token not found. Please set the HF_TOKEN environment variable.")
    st.stop()

client = InferenceClient(token=HF_TOKEN)

# --- Image Generation ---
def generate_landscaped_image(image, prompt):
    """Generates a landscaped image using the Hugging Face Inference API."""
    try:
        # Ensure image is large enough (minimum 512x512 pixels)
        min_size = 512
        if min(image.size) < min_size:
            # Resize image to minimum size while maintaining aspect ratio
            ratio = max(image.size[0] / min_size, image.size[1] / min_size)
            new_size = (int(image.size[0] / ratio), int(image.size[1] / ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            st.write(f"Resized image to: {new_size}")

        # Convert image to RGB if it's not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # The API expects the image as bytes
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=95)
        img_bytes = buffered.getvalue()
        
        # Log the image size and format for debugging
        st.write(f"Image size: {image.size}")
        st.write(f"Image format: {image.format}")
        st.write(f"Image mode: {image.mode}")
        st.write(f"Image bytes length: {len(img_bytes)}")

        # Try the model with additional parameters
        result = client.image_to_image(
            image=img_bytes,
            prompt=prompt,
            model="runwayml/stable-diffusion-v1-5",
            strength=0.8,
            guidance_scale=7.5,
            num_inference_steps=50,  # Add number of steps
            seed=42  # Add seed for reproducibility
        )
        
        # Check if we got a valid response
        if not result:
            st.error("No response from API")
            return None

        # Try to convert the result to PIL Image
        try:
            # First try to decode as base64
            try:
                import base64
                result = base64.b64decode(result)
            except:
                pass

            # Try to open the image
            result_image = Image.open(io.BytesIO(result))
            return result_image
        except Exception as e:
            st.error(f"Failed to convert API response to image: {e}")
            st.write(f"Response type: {type(result)}")
            st.write(f"Response length: {len(result) if hasattr(result, '__len__') else 'N/A'}")
            return None
    
    except Exception as e:
        st.error(f"An error occurred during image generation: {e}")
        st.write(f"Error type: {type(e)}")
        st.write(f"Error details: {str(e)}")
        return None

# --- UI Components ---
def load_sample_images():
    before = Image.open(os.path.join("samples", "x.jpeg"))
    after = Image.open(os.path.join("samples", "y.jpeg"))
    return before, after

def render_before_after_carousel(before_image, after_image):
    def pil_to_base64(img):
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    before_b64 = pil_to_base64(before_image)
    after_b64 = pil_to_base64(after_image)
    
    html_code = f"""
    <style>
    .container {{ position: relative; width: 100%; max-width: 700px; }}
    .img-comp-container {{ position: relative; height: auto; }}
    .img-comp-img {{ position: absolute; width: auto; height: auto; overflow: hidden; }}
    .img-comp-slider {{ position: absolute; z-index: 9; cursor: ew-resize; width: 5px; height: 100%; background-color: #fff; opacity: 0.7; }}
    </style>
    <div class="container">
        <div class="img-comp-container" id="slider-container">
            <img src="data:image/png;base64,{before_b64}" style="width: 100%; position: relative;">
            <div style="position: absolute; top: 0; left: 0; overflow: hidden; width: 50%;">
                <img src="data:image/png;base64,{after_b64}" style="width: 100%;">
            </div>
        </div>
    </div>
    <script>
    document.addEventListener("DOMContentLoaded", function () {{
        const container = document.getElementById("slider-container");
        const topImgWrapper = container.children[1];
        const slider = document.createElement("div");
        slider.classList.add("img-comp-slider");
        container.appendChild(slider);
        slider.style.left = "50%";

        let clicked = false;
        slider.addEventListener("mousedown", () => clicked = true);
        window.addEventListener("mouseup", () => clicked = false);
        window.addEventListener("mousemove", e => {{
            if (!clicked) return;
            const rect = container.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const width = Math.max(0, Math.min(x, container.offsetWidth));
            topImgWrapper.style.width = width + "px";
            slider.style.left = width + "px";
        }});
    }});
    </script>
    """
    components.html(html_code, height=500)

# --- Main Application ---
st.set_page_config(page_title="LandscapeAI", layout="centered")
st.title("🌿 AI Landscaping Demo")

uploaded_file = st.file_uploader("📸 Upload a photo of your yard", type=["jpg", "jpeg", "png"])
prompt = st.text_area("📝 Describe your landscaping idea", placeholder="e.g. Add flower beds and a stone walkway")

if uploaded_file and prompt:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_container_width=True)
    if st.button("🚀 Generate Landscaping"):
        with st.spinner("Creating your new landscape... This may take a moment."):
            result_image = generate_landscaped_image(image, prompt)
            if result_image:
                st.subheader("🪄 Before & After Comparison")
                render_before_after_carousel(image, result_image)
            else:
                st.error("❌ AI generation failed.")
else:
    st.info("📸 No image uploaded. Showing a sample transformation:")
    before_img, after_img = load_sample_images()
    render_before_after_carousel(before_img, after_img)
