import streamlit as st
from PIL import Image
import tempfile
import torch
import os
import sys
import cv2
import numpy as np
import pathlib
from pathlib import Path
import base64


# Function to load and encode local jpg image
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Local image filename (same folder)
image_file = 'fff.jpg'

# Get base64 string
img_base64 = get_base64_of_bin_file(image_file)

# Inject HTML + CSS for background
page_bg_img = f"""
<style>
.stApp {{
  background-image: url("data:image/jpg;base64,{img_base64}");
  background-size: cover;
  background-repeat: no-repeat;
  background-attachment: fixed;
}}
</style>
"""

# Load CSS
st.markdown(page_bg_img, unsafe_allow_html=True)


# --- Page Definitions ---
def page1():
    st.markdown('<h3 style="color:white;">Welcome to the App</h3>', unsafe_allow_html=True)
    if st.button("Go to Prediction Page"):
        return "page2"
    return "page1"

def page2():
    #st.markdown('<h3 style="color:white;">Prediction Page</h3>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back"):
            return "page1"
    with col2:
        if st.button("Go to Creator Info"):
            return "page3"
    return "page2"

def page3():
    #st.markdown('<h3 style="color:white;">👨‍💻 Creator Info</h3>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back"):
            return "page2"
    return "page3"



# Initialize page
if "current_page" not in st.session_state:
    st.session_state.current_page = "page1"

# Render pages
if st.session_state.current_page == "page1":
    st.markdown("""
    <h1 style='color: white;'>
        🛡️ Military Soldier Safety and Weapon Detection using YOLO
    </h1> """, unsafe_allow_html=True)
    
    st.markdown(""" \n \n  \n""")
    st.markdown(""" \n \n  \n""")
    
    st.markdown(
    """
    <h4 style="color:white;">
        This project aims to address these challenges by leveraging computer vision and 
        YOLO (You Only Look Once), a state-of-the-art object detection algorithm, to 
        automate the process of detecting and classifying objects in real-time.
    </h4>
    """, unsafe_allow_html=True)


    st.markdown(""" \n \n""")
    st.markdown(""" <style>.image-right { float : left; } </style> """, unsafe_allow_html = True, )
    st.markdown('<div class = "image-left">', unsafe_allow_html = True)
    st.image("aaa.jpeg", width=205)
    st.markdown('</div>', unsafe_allow_html = True)
    st.markdown(""" \n \n""")
    
    st.markdown(
    """
    <h3 style="color:white;">Skills Take Away From This Project</h3>
    <p style="color:white;">
        • Python<br>
        • OpenCV<br>
        • Streamlit<br>
        • Deep Learning: YOLO object detection model.<br>
        • Computer Vision: OpenCV for preprocessing and visualization.<br>
        • OCR Technology: Tesseract/EasyOCR for text recognition.<br>
        • Web Application: Streamlit for interactive deployment.<br>
        • Python Libraries: NumPy, Pandas, Matplotlib, Seaborn, TensorFlow/PyTorch.<br>
        • Image Processing: Includes resizing, contrast enhancement, and edge detection techniques.
    </p>
    """, unsafe_allow_html=True)

    st.markdown(""" \n \n""")

    next_page = page1()
    
    if next_page == "page2":
        st.session_state.current_page = "page2"
        
elif st.session_state.current_page == "page2":

    # Set up paths
    sys.path.append('yolov5')
    pathlib.PosixPath = pathlib.WindowsPath
    
    # Import YOLOv5 modules
    from models.common import DetectMultiBackend
    from utils.general import non_max_suppression
    from utils.augmentations import letterbox
    
    # Streamlit UI
    st.set_page_config(page_title="Military Object Detection", layout="centered")
    st.markdown("""
        <h1 style='color: white;'>🎯 Military Object Detection using YOLOv5</h1>
    """, unsafe_allow_html=True)
    
    
    # Load model
    @st.cache_resource
    def load_model():
        return DetectMultiBackend('best.pt', device='cpu')
    
    model = load_model()
    class_names = model.names  # List of class names (0 to 11)
    
    # Upload image
    st.markdown("<p style='color: white; font-size:18px;'>📤 Upload an image</p>", unsafe_allow_html=True)
    # File uploader with label hidden
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        # Save file to temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file.read())
            temp_img_path = tmp.name
    
        # Load and preprocess
        img0 = cv2.imread(temp_img_path)
        img = letterbox(img0, 640, stride=32, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img_tensor = torch.from_numpy(img).to('cpu').float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)
    
        # Inference
        pred = model(img_tensor, augment=False)
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]
    
        # Draw boxes
        if pred is not None and len(pred):
            for *xyxy, conf, cls in pred:
                label = f"{class_names[int(cls)]} {conf:.2f}"
                p1 = (int(xyxy[0]), int(xyxy[1]))
                p2 = (int(xyxy[2]), int(xyxy[3]))
                cv2.rectangle(img0, p1, p2, (0, 255, 0), 2)
                cv2.putText(img0, label, (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
        # Show results
        #st.subheader("✅ Detection complete")
    
        st.markdown("""
            <div style="
                background-color: #28a745;
                padding: 15px;
                border-radius: 10px;
                color: white;
                font-size: 18px;
                font-weight: bold;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
            ">
                ✅ Detection complete
            </div>
        """, unsafe_allow_html=True)
    
        st.image(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB), caption="Detected Objects", use_container_width=True)

    next_page = page2()
    
    if next_page == "page1":
        st.session_state.current_page = "page1"
    elif next_page == "page3":
        st.session_state.current_page = "page3"

elif st.session_state.current_page == "page3":
    
    st.markdown(
    "<h1 style='color:white;'>💻 Creator of this Project</h1>",
    unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown(
    """
    <h2 style="color:white;"><strong>Developed by:</strong>     Sudharsan M S 👨‍💻</h2>
    <h3 style="color:white;"><strong>Skills:</strong> Python 🐍, OpenCV 📈, Deep Learning 🧠, Computer Vision 👀, Streamlit ⌛</h3>
    """, unsafe_allow_html=True)


    st.image('asd.jpg', width=190)

    next_page = page3()
    if next_page == "page2":
        st.session_state.current_page = "page2"
        