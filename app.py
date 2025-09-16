import streamlit as st
import cv2
import torch
import torchvision
from ultralytics import YOLO
import numpy as np
import time
from PIL import Image


st.set_page_config(
    page_title="Object Detection Comparison",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Comparing YOLOv8, SSD, and Faster R-CNN 🚀")
st.write("Upload an image or use your webcam to see a real-time comparison of three popular object detection models.")

@st.cache_resource
def load_yolo_model():
    model = YOLO('yolov8n.pt')
    return model

@st.cache_resource
def load_ssd_model():
    model = torchvision.models.detection.ssd300_vgg16(weights='SSD300_VGG16_Weights.DEFAULT')
    model.eval()
    return model

@st.cache_resource
def load_faster_rcnn_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.DEFAULT')
    model.eval()
    return model

with st.spinner('Loading models... This may take a moment.'):
    yolo_model = load_yolo_model()
    ssd_model = load_ssd_model()
    faster_rcnn_model = load_faster_rcnn_model()


COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def draw_boxes(image, boxes, labels, scores, confidence_threshold):
    image = np.array(image)
    for i, box in enumerate(boxes):
        if scores[i] > confidence_threshold:
            x1, y1, x2, y2 = map(int, box)
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Prepare text
            label_text = f"{labels[i]}: {scores[i]:.2f}"
            # Put text
            cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return image

def process_image(image, confidence_threshold):
    results = {}
    original_image = np.array(image)
 
    start_time = time.time()
    yolo_preds = yolo_model(original_image)
    yolo_boxes = yolo_preds[0].boxes.xyxy.cpu().numpy()
    yolo_scores = yolo_preds[0].boxes.conf.cpu().numpy()
    yolo_classes = yolo_preds[0].boxes.cls.cpu().numpy()
    yolo_labels = [yolo_model.names[int(c)] for c in yolo_classes]
    yolo_time = time.time() - start_time
    yolo_image = draw_boxes(image, yolo_boxes, yolo_labels, yolo_scores, confidence_threshold)
    results['YOLOv8'] = (yolo_image, yolo_time)

    frame_tensor = torchvision.transforms.functional.to_tensor(original_image).unsqueeze(0)

    start_time = time.time()
    with torch.no_grad():
        ssd_preds = ssd_model(frame_tensor)[0]
    ssd_boxes = ssd_preds['boxes'].cpu().numpy()
    ssd_scores = ssd_preds['scores'].cpu().numpy()
    ssd_classes = ssd_preds['labels'].cpu().numpy()
    ssd_labels = [COCO_CLASSES[c] for c in ssd_classes]
    ssd_time = time.time() - start_time
    ssd_image = draw_boxes(image, ssd_boxes, ssd_labels, ssd_scores, confidence_threshold)
    results['SSD'] = (ssd_image, ssd_time)

    start_time = time.time()
    with torch.no_grad():
        frcnn_preds = faster_rcnn_model(frame_tensor)[0]
    frcnn_boxes = frcnn_preds['boxes'].cpu().numpy()
    frcnn_scores = frcnn_preds['scores'].cpu().numpy()
    frcnn_classes = frcnn_preds['labels'].cpu().numpy()
    frcnn_labels = [COCO_CLASSES[c] for c in frcnn_classes]
    frcnn_time = time.time() - start_time
    frcnn_image = draw_boxes(image, frcnn_boxes, frcnn_labels, frcnn_scores, confidence_threshold)
    results['Faster R-CNN'] = (frcnn_image, frcnn_time)
    
    return results

st.sidebar.header("Configuration")
confidence_slider = st.sidebar.slider(
    "Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05
)

input_option = st.sidebar.radio(
    "Choose your input method:",
    ("Upload an Image", "Use Webcam (Coming Soon)"), # Webcam support requires streamlit-webrtc and is more complex. Starting with image upload.
)

if input_option == "Upload an Image":
    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        
        st.image(image, caption="Original Image", use_container_width=True)
        st.markdown("---")
        
        with st.spinner("Processing image..."):
            detection_results = process_image(image, confidence_slider)
        
        st.success("Processing Complete!")

        # Display results in three columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.header("YOLOv8")
            img, exec_time = detection_results['YOLOv8']
            st.image(img, caption=f"Processed by YOLOv8 in {exec_time:.2f}s", use_container_width=True)
            
        with col2:
            st.header("SSD")
            img, exec_time = detection_results['SSD']
            st.image(img, caption=f"Processed by SSD in {exec_time:.2f}s", use_container_width=True)

        with col3:
            st.header("Faster R-CNN")
            img, exec_time = detection_results['Faster R-CNN']
            st.image(img, caption=f"Processed by Faster R-CNN in {exec_time:.2f}s", use_container_width=True)

elif input_option == "Use Webcam (Coming Soon)":
    st.info("Webcam functionality is under development. Please use the image upload feature for now.")
    st.warning("Real-time detection on a public cloud server can be slow due to network latency and processing power limitations.")
