import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os
import json
from datetime import datetime
import time
import cv2
from PIL import Image, ImageDraw, ImageFont
import glob
import torch
import math

# Set page config first - MUST be the first Streamlit command
st.set_page_config(layout="wide", page_title="YOLO Stereo Vision Analysis")

# Import functions from your main file
# Note: Assuming your main file is named yolo_stereo_detection.py
# You may need to adjust this import based on your actual file structure
try:
    from webapp import load_model, calculate_depth, calculate_speed, infer_image
except ImportError:
    # Simplified versions for standalone use
    def load_model(device, model_type='yolov5s'):
        try:
            model = torch.hub.load('ultralytics/yolov5', model_type, pretrained=True)
            model.to(device)
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
            
    def calculate_depth(x1, x2):
        f = 700  # Focal length
        B = 0.1  # Baseline
        if x1 == x2:
            return 1000.0
        Z = (f * B) / abs(x1 - x2)
        return Z
        
    def calculate_speed(position1, position2, time_interval):
        x1_1, y1_1, x2_1, y2_1, t1 = position1
        x1_2, y1_2, x2_2, y2_2, t2 = position2
        
        if time_interval <= 0.001:
            time_interval = 0.001
            
        try:
            Z1 = calculate_depth(x1_1, x2_1)
            Z2 = calculate_depth(x1_2, x2_2)
            distance = math.sqrt((x1_2 - x1_1)**2 + (y1_2 - y1_1)**2 + (Z2 - Z1)**2)
            speed = distance / time_interval
            return speed
        except Exception as e:
            return 0.0

# Initialize session state for data
if 'detection_data' not in st.session_state:
    st.session_state.detection_data = []
if 'frame_positions' not in st.session_state:
    st.session_state.frame_positions = []
if 'timestamps' not in st.session_state:
    st.session_state.timestamps = []
if 'model' not in st.session_state:
    st.session_state.model = None
if 'confidence' not in st.session_state:
    st.session_state.confidence = 0.45

def process_frames(left_frames, right_frames):
    """Process a pair of frames and extract detection data"""
    if st.session_state.model is None:
        st.error("Model not loaded. Please load the model first.")
        return
        
    model = st.session_state.model
    confidence = st.session_state.confidence
    model.conf = confidence
    
    current_time = time.time()
    st.session_state.timestamps.append(current_time)
    
    # Process left frame
    left_result = model(left_frames)
    left_positions = []
    
    if len(left_result.pred) > 0 and len(left_result.pred[0]) > 0:
        for det in left_result.pred[0]:
            x1, y1, x2, y2 = map(float, det[:4].cpu().numpy())
            conf = float(det[4].cpu().numpy())
            cls_id = int(det[5].cpu().numpy())
            cls_name = model.names[cls_id]
            
            left_positions.append({
                "x1": x1, 
                "y1": y1, 
                "x2": x2, 
                "y2": y2, 
                "conf": conf,
                "class_id": cls_id,
                "class_name": cls_name
            })
    
    # Process right frame
    right_result = model(right_frames)
    right_positions = []
    
    if len(right_result.pred) > 0 and len(right_result.pred[0]) > 0:
        for det in right_result.pred[0]:
            x1, y1, x2, y2 = map(float, det[:4].cpu().numpy())
            conf = float(det[4].cpu().numpy())
            cls_id = int(det[5].cpu().numpy())
            cls_name = model.names[cls_id]
            
            right_positions.append({
                "x1": x1, 
                "y1": y1, 
                "x2": x2, 
                "y2": y2, 
                "conf": conf,
                "class_id": cls_id,
                "class_name": cls_name
            })
    
    # Store positions for later use in speed calculation
    st.session_state.frame_positions.append({
        "left": left_positions,
        "right": right_positions,
        "timestamp": current_time
    })
    
    # Calculate depths and speeds if we have at least 2 frames
    if len(st.session_state.frame_positions) >= 2:
        prev_frame = st.session_state.frame_positions[-2]
        curr_frame = st.session_state.frame_positions[-1]
        time_interval = curr_frame["timestamp"] - prev_frame["timestamp"]
        
        # Match objects between frames (simple approach - can be improved)
        for i, left_obj in enumerate(curr_frame["left"]):
            # Find matching object in right frame (simple approach using class and proximity)
            best_match_right = None
            best_match_dist = float('inf')
            
            for right_obj in curr_frame["right"]:
                if right_obj["class_id"] == left_obj["class_id"]:
                    # Calculate center points
                    left_center_x = (left_obj["x1"] + left_obj["x2"]) / 2
                    left_center_y = (left_obj["y1"] + left_obj["y2"]) / 2
                    right_center_x = (right_obj["x1"] + right_obj["x2"]) / 2
                    right_center_y = (right_obj["y1"] + right_obj["y2"]) / 2
                    
                    # Calculate distance between centers (emphasize y-coordinate for stereo matching)
                    dist = math.sqrt((left_center_x - right_center_x)**2 + 3*(left_center_y - right_center_y)**2)
                    
                    if dist < best_match_dist:
                        best_match_dist = dist
                        best_match_right = right_obj
            
            if best_match_right and best_match_dist < 100:  # Threshold for matching
                # Calculate depth
                left_center_x = (left_obj["x1"] + left_obj["x2"]) / 2
                right_center_x = (best_match_right["x1"] + best_match_right["x2"]) / 2
                depth = calculate_depth(left_center_x, right_center_x)
                
                # Find matching object in previous frame
                best_prev_match = None
                best_prev_dist = float('inf')
                
                for prev_left_obj in prev_frame["left"]:
                    if prev_left_obj["class_id"] == left_obj["class_id"]:
                        prev_center_x = (prev_left_obj["x1"] + prev_left_obj["x2"]) / 2
                        prev_center_y = (prev_left_obj["y1"] + prev_left_obj["y2"]) / 2
                        curr_center_x = (left_obj["x1"] + left_obj["x2"]) / 2
                        curr_center_y = (left_obj["y1"] + left_obj["y2"]) / 2
                        
                        dist = math.sqrt((prev_center_x - curr_center_x)**2 + (prev_center_y - curr_center_y)**2)
                        
                        if dist < best_prev_dist:
                            best_prev_dist = dist
                            best_prev_match = prev_left_obj
                
                # Calculate speed if we have a match in previous frame
                speed = 0
                if best_prev_match and best_prev_dist < 100:
                    # Format: [x1, y1, x2, y2, timestamp]
                    pos1 = [best_prev_match["x1"], best_prev_match["y1"], 
                           best_prev_match["x2"], best_prev_match["y2"], 
                           prev_frame["timestamp"]]
                    pos2 = [left_obj["x1"], left_obj["y1"], 
                           left_obj["x2"], left_obj["y2"], 
                           curr_frame["timestamp"]]
                    
                    speed = calculate_speed(pos1, pos2, time_interval) * 0.8  # Scale factor as in original code
                
                # Store detection data
                detection_data = {
                    "frame_id": len(st.session_state.detection_data),
                    "timestamp": curr_frame["timestamp"],
                    "class_id": left_obj["class_id"],
                    "class_name": left_obj["class_name"],
                    "confidence": left_obj["conf"],
                    "left_x1": left_obj["x1"],
                    "left_y1": left_obj["y1"],
                    "left_x2": left_obj["x2"],
                    "left_y2": left_obj["y2"],
                    "right_x1": best_match_right["x1"],
                    "right_y1": best_match_right["y1"],
                    "right_x2": best_match_right["x2"],
                    "right_y2": best_match_right["y2"],
                    "depth": depth,
                    "speed": speed,
                    "width": left_obj["x2"] - left_obj["x1"],
                    "height": left_obj["y2"] - left_obj["y1"]
                }
                
                st.session_state.detection_data.append(detection_data)

def plot_speed_over_time():
    """Plot speed of detected objects over time"""
    if not st.session_state.detection_data:
        st.warning("No speed data available.")
        return
        
    df = pd.DataFrame(st.session_state.detection_data)
    
    if len(df) < 2:
        st.warning("Insufficient data for speed plot. Need at least 2 detection points.")
        return
    
    fig = px.line(df, x="frame_id", y="speed", color="class_name",
                 title="Object Speed Over Time",
                 labels={"frame_id": "Frame Number", "speed": "Speed (km/s)", "class_name": "Object Class"})
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def plot_depth_distribution():
    """Plot histogram of depth distribution"""
    if not st.session_state.detection_data:
        st.warning("No depth data available.")
        return
        
    df = pd.DataFrame(st.session_state.detection_data)
    
    fig = px.histogram(df, x="depth", color="class_name", 
                      title="Distribution of Object Depths",
                      labels={"depth": "Depth (meters)", "count": "Frequency", "class_name": "Object Class"},
                      nbins=20)
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def plot_object_counts():
    """Plot counts of different detected objects"""
    if not st.session_state.detection_data:
        st.warning("No detection data available.")
        return
        
    df = pd.DataFrame(st.session_state.detection_data)
    counts = df["class_name"].value_counts().reset_index()
    counts.columns = ["Object", "Count"]
    
    fig = px.bar(counts, x="Object", y="Count", 
                title="Detected Object Counts",
                color="Object")
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def plot_confidence_distribution():
    """Plot distribution of detection confidence scores"""
    if not st.session_state.detection_data:
        st.warning("No detection data available.")
        return
        
    df = pd.DataFrame(st.session_state.detection_data)
    
    fig = px.histogram(df, x="confidence", color="class_name",
                      title="Distribution of Confidence Scores",
                      labels={"confidence": "Confidence Score", "count": "Frequency", "class_name": "Object Class"},
                      nbins=20)
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def plot_speed_vs_depth():
    """Plot speed vs depth scatter plot"""
    if not st.session_state.detection_data:
        st.warning("No detection data available.")
        return
        
    df = pd.DataFrame(st.session_state.detection_data)
    
    if len(df) < 2:
        st.warning("Insufficient data for speed vs depth plot. Need at least 2 detection points.")
        return
    
    fig = px.scatter(df, x="depth", y="speed", color="class_name", 
                    size="confidence", hover_data=["frame_id"],
                    title="Speed vs. Depth by Object Class",
                    labels={"depth": "Depth (meters)", "speed": "Speed (km/s)", "class_name": "Object Class"})
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def plot_object_size_vs_depth():
    """Plot object size vs depth to visualize perspective scaling"""
    if not st.session_state.detection_data:
        st.warning("No detection data available.")
        return
        
    df = pd.DataFrame(st.session_state.detection_data)
    df["area"] = df["width"] * df["height"]
    
    fig = px.scatter(df, x="depth", y="area", color="class_name",
                    title="Object Size vs. Depth",
                    labels={"depth": "Depth (meters)", "area": "Object Area (pixels²)", "class_name": "Object Class"})
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def analyze_video_pair():
    """Process and analyze a pair of stereo videos"""
    vid_fileL = st.session_state.get('vid_fileL')
    vid_fileR = st.session_state.get('vid_fileR')
    
    if not vid_fileL or not vid_fileR:
        st.warning("Please select both left and right video files.")
        return
    
    if not os.path.exists(vid_fileL) or not os.path.exists(vid_fileR):
        st.error("Video files not found. Please check the paths.")
        return
        
    # Reset detection data for new analysis
    st.session_state.detection_data = []
    st.session_state.frame_positions = []
    st.session_state.timestamps = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Open video files
    capL = cv2.VideoCapture(vid_fileL)
    capR = cv2.VideoCapture(vid_fileR)
    
    if not capL.isOpened() or not capR.isOpened():
        st.error("Error opening video files. Please check the format.")
        return
    
    # Get video properties
    total_frames = int(min(capL.get(cv2.CAP_PROP_FRAME_COUNT), capR.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = capL.get(cv2.CAP_PROP_FPS)
    
    # Determine how many frames to analyze based on the total
    if total_frames > 300:
        # For long videos, analyze every nth frame to keep processing time reasonable
        frame_interval = total_frames // 100
    else:
        frame_interval = 1
    
    # Process frames
    frame_count = 0
    while True:
        retL, frameL = capL.read()
        retR, frameR = capR.read()
        
        if not retL or not retR or frame_count >= total_frames:
            break
            
        # Only process every Nth frame to speed up analysis
        if frame_count % frame_interval == 0:
            # Update progress
            progress = min(frame_count / total_frames, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {frame_count}/{total_frames} ({progress*100:.1f}%)")
            
            # Resize frames to standard size for processing
            frameL = cv2.resize(frameL, (640, 480))
            frameL = cv2.cvtColor(frameL, cv2.COLOR_BGR2RGB)
            
            frameR = cv2.resize(frameR, (640, 480))
            frameR = cv2.cvtColor(frameR, cv2.COLOR_BGR2RGB)
            
            # Process the pair of frames
            process_frames(frameL, frameR)
        
        frame_count += 1
    
    # Release video resources
    capL.release()
    capR.release()
    
    # Complete the progress bar
    progress_bar.progress(1.0)
    status_text.text(f"Analysis complete! Processed {frame_count} frames and detected {len(st.session_state.detection_data)} objects.")
    
    # Display result summary
    st.success(f"Video analysis complete. Detected {len(st.session_state.detection_data)} objects across {frame_count} frames.")

def analyze_image_pair():
    """Process and analyze a pair of stereo images"""
    img_fileL = st.session_state.get('img_fileL')
    img_fileR = st.session_state.get('img_fileR')
    
    if not img_fileL or not img_fileR:
        st.warning("Please select both left and right image files.")
        return
    
    if not os.path.exists(img_fileL) or not os.path.exists(img_fileR):
        st.error("Image files not found. Please check the paths.")
        return
    
    # Reset detection data for new analysis
    st.session_state.detection_data = []
    st.session_state.frame_positions = []
    st.session_state.timestamps = []
    
    # Load images
    try:
        imgL = cv2.imread(img_fileL)
        imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
        imgL = cv2.resize(imgL, (640, 480))
        
        imgR = cv2.imread(img_fileR)
        imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB)
        imgR = cv2.resize(imgR, (640, 480))
    except Exception as e:
        st.error(f"Error loading images: {e}")
        return
    
    # Process the image pair
    process_frames(imgL, imgR)
    
    # Display result summary
    st.success(f"Image analysis complete. Detected {len(st.session_state.detection_data)} objects.")

def display_data_table():
    """Display raw detection data in a table"""
    if not st.session_state.detection_data:
        st.warning("No detection data available.")
        return
        
    df = pd.DataFrame(st.session_state.detection_data)
    
    # Only show relevant columns in a more readable format
    if len(df) > 0:
        display_cols = ["frame_id", "class_name", "confidence", "speed", "depth"]
        if all(col in df.columns for col in display_cols):
            display_df = df[display_cols].copy()
            
            # Format numbers for better readability
            display_df["confidence"] = display_df["confidence"].map(lambda x: f"{x:.2f}" if x is not None else "N/A")
            display_df["speed"] = display_df["speed"].map(lambda x: f"{x:.2f}" if x is not None else "N/A")
            display_df["depth"] = display_df["depth"].map(lambda x: f"{x:.2f}" if x is not None else "N/A")
            
            st.dataframe(display_df, use_container_width=True)
        else:
            st.dataframe(df, use_container_width=True)
    else:
        st.warning("No data to display.")

def calculate_statistics():
    """Calculate and display key statistics from the detection data"""
    if not st.session_state.detection_data:
        st.warning("No detection data available.")
        return
        
    df = pd.DataFrame(st.session_state.detection_data)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Detections", len(df))
        if "class_name" in df.columns:
            st.metric("Unique Object Classes", len(df["class_name"].unique()))
        
    with col2:
        if "speed" in df.columns and len(df) > 0:
            avg_speed = df["speed"].mean()
            max_speed = df["speed"].max()
            st.metric("Average Speed (km/s)", f"{avg_speed:.2f}")
            st.metric("Maximum Speed (km/s)", f"{max_speed:.2f}")
        
    with col3:
        if "depth" in df.columns and len(df) > 0:
            avg_depth = df["depth"].mean()
            max_depth = df["depth"].max()
            st.metric("Average Depth (m)", f"{avg_depth:.2f}")
            st.metric("Maximum Depth (m)", f"{max_depth:.2f}")
    
    # Calculate and display speed statistics by object class
    if "class_name" in df.columns and "speed" in df.columns and "depth" in df.columns and len(df["class_name"].unique()) > 1:
        st.subheader("Statistics by Object Class")
        
        stats_df = df.groupby("class_name").agg({
            "speed": ["mean", "max", "count"],
            "depth": ["mean", "max"],
            "confidence": ["mean"]
        }).reset_index()
        
        # Flatten the MultiIndex columns
        stats_df.columns = [' '.join(col).strip() for col in stats_df.columns.values]
        stats_df = stats_df.rename(columns={
            "class_name ": "Object Class",
            "speed mean": "Avg Speed (km/s)",
            "speed max": "Max Speed (km/s)",
            "speed count": "Count",
            "depth mean": "Avg Depth (m)",
            "depth max": "Max Depth (m)",
            "confidence mean": "Avg Confidence"
        })
        
        # Format the numbers
        for col in ["Avg Speed (km/s)", "Max Speed (km/s)", "Avg Depth (m)", "Max Depth (m)", "Avg Confidence"]:
            if col in stats_df.columns:
                stats_df[col] = stats_df[col].map(lambda x: f"{x:.2f}" if x is not None else "N/A")
        
        st.dataframe(stats_df, use_container_width=True)

def save_results():
    """Save analysis results to a file"""
    if not st.session_state.detection_data:
        st.warning("No data to save.")
        return
        
    # Create directory if it doesn't exist
    os.makedirs("data/analysis_results", exist_ok=True)
    
    # Create a timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Convert data to DataFrame
    df = pd.DataFrame(st.session_state.detection_data)
    
    # Save as CSV
    csv_path = f"data/analysis_results/detection_results_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    
    # Also save as JSON for complete data preservation
    json_path = f"data/analysis_results/detection_results_{timestamp}.json"
    df.to_json(json_path, orient="records")
    
    st.success(f"Results saved to:\n- {csv_path}\n- {json_path}")
    
    # Provide download links
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name=f"detection_results_{timestamp}.csv",
        mime="text/csv"
    )

def main():
    st.title("YOLO Stereo-Camera Visualization Dashboard")
    
    # Sidebar for input and configuration
    st.sidebar.title("Input & Configuration")
    
    # Model selection
    model_type = st.sidebar.selectbox("Select Model Type", ['yolov5s', 'yolov5m', 'yolov5l', 'yolov5x'])
    confidence = st.sidebar.slider('Confidence Threshold', min_value=0.1, max_value=1.0, value=0.45)
    
    # Load model button
    if st.sidebar.button("Load Model"):
        with st.spinner("Loading model..."):
            st.session_state.model = load_model("cpu", model_type=model_type)
            st.session_state.confidence = confidence
        st.sidebar.success(f"Model {model_type} loaded successfully!")
    
    # Input type selection
    input_option = st.sidebar.radio("Select Input Type:", ['Image Pair', 'Video Pair'])
    data_src = st.sidebar.radio("Select Input Source:", ['Sample', 'Upload Files'])
    
    # Handle file input based on selection
    if input_option == 'Image Pair':
        if data_src == 'Sample':
            # Use sample images if available
            sample_dir = "data/sample_images"
            if os.path.exists(sample_dir):
                left_samples = glob.glob(f"{sample_dir}/left/*")
                right_samples = glob.glob(f"{sample_dir}/right/*")
                
                if left_samples and right_samples:
                    st.session_state.img_fileL = left_samples[0]
                    st.session_state.img_fileR = right_samples[0]
                else:
                    st.sidebar.warning("No sample images found. Please use upload option.")
            else:
                st.sidebar.warning("Sample directory not found. Please use upload option.")
        else:
            # File uploaders for images
            left_img = st.sidebar.file_uploader("Upload Left Image", type=['png', 'jpg', 'jpeg'])
            right_img = st.sidebar.file_uploader("Upload Right Image", type=['png', 'jpg', 'jpeg'])
            
            if left_img:
                os.makedirs("data/uploaded_data", exist_ok=True)
                img_fileL = f"data/uploaded_data/uploadL.{left_img.name.split('.')[-1]}"
                with open(img_fileL, 'wb') as f:
                    f.write(left_img.getbuffer())
                st.session_state.img_fileL = img_fileL
            
            if right_img:
                os.makedirs("data/uploaded_data", exist_ok=True)
                img_fileR = f"data/uploaded_data/uploadR.{right_img.name.split('.')[-1]}"
                with open(img_fileR, 'wb') as f:
                    f.write(right_img.getbuffer())
                st.session_state.img_fileR = img_fileR
        
        # Analyze button for images
        if st.sidebar.button("Analyze Images"):
            if st.session_state.model is None:
                st.error("Please load the model first.")
            else:
                with st.spinner("Analyzing images..."):
                    analyze_image_pair()
    
    else:  # Video Pair
        if data_src == 'Sample':
            # Use sample videos if available
            sample_dir = "data/sample_videos"
            if os.path.exists(sample_dir):
                left_samples = glob.glob(f"{sample_dir}/left/*")
                right_samples = glob.glob(f"{sample_dir}/right/*")
                
                if left_samples and right_samples:
                    st.session_state.vid_fileL = left_samples[0]
                    st.session_state.vid_fileR = right_samples[0]
                else:
                    st.sidebar.warning("No sample videos found. Please use upload option.")
            else:
                st.sidebar.warning("Sample directory not found. Please use upload option.")
        else:
            # File uploaders for videos
            left_vid = st.sidebar.file_uploader("Upload Left Video", type=['mp4', 'avi', 'mov'])
            right_vid = st.sidebar.file_uploader("Upload Right Video", type=['mp4', 'avi', 'mov'])
            
            if left_vid:
                os.makedirs("data/uploaded_data", exist_ok=True)
                vid_fileL = f"data/uploaded_data/uploadL.{left_vid.name.split('.')[-1]}"
                with open(vid_fileL, 'wb') as f:
                    f.write(left_vid.getbuffer())
                st.session_state.vid_fileL = vid_fileL
            
            if right_vid:
                os.makedirs("data/uploaded_data", exist_ok=True)
                vid_fileR = f"data/uploaded_data/uploadR.{right_vid.name.split('.')[-1]}"
                with open(vid_fileR, 'wb') as f:
                    f.write(right_vid.getbuffer())
                st.session_state.vid_fileR = vid_fileR
        
        # Analyze button for videos
        if st.sidebar.button("Analyze Videos"):
            if st.session_state.model is None:
                st.error("Please load the model first.")
            else:
                with st.spinner("Analyzing videos... This may take some time."):
                    analyze_video_pair()
    
    # Save results button
    if st.session_state.detection_data:
        st.sidebar.button("Save Results", on_click=save_results)
    
    # Create tabs for different visualization categories
    if st.session_state.detection_data:
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Speed Analysis", "Object Analysis", "Raw Data"])
        
        with tab1:
            st.header("Detection Overview")
            
            # Display key statistics
            calculate_statistics()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # In tab1 (Overview), col1
                plot_object_counts()
            
            with col2:
                plot_confidence_distribution()
            
            # Display depth distribution
            st.subheader("Depth Distribution")
            plot_depth_distribution()
        
        with tab2:
            st.header("Speed Analysis")
            
            # Display speed over time
            st.subheader("Speed Over Time")
            plot_speed_over_time()
            
            # Display speed vs depth
            st.subheader("Speed vs. Depth")
            plot_speed_vs_depth()
            
            # Additional speed analysis
            if len(st.session_state.detection_data) > 0:
                df = pd.DataFrame(st.session_state.detection_data)
                
                # Show speed statistics by object class
                st.subheader("Speed Statistics by Object Class")
                if "speed" in df.columns and "class_name" in df.columns:
                    speed_stats = df.groupby("class_name")["speed"].agg(['mean', 'min', 'max']).reset_index()
                    speed_stats = speed_stats.rename(columns={'mean': 'Average Speed', 'min': 'Min Speed', 'max': 'Max Speed'})
                    
                    # Format the numbers
                    for col in ["Average Speed", "Min Speed", "Max Speed"]:
                        speed_stats[col] = speed_stats[col].map(lambda x: f"{x:.2f} km/s")
                    
                    st.dataframe(speed_stats, use_container_width=True)
                    
                    # Create a box plot of speeds by class
                    fig = px.box(df, x="class_name", y="speed", 
                                title="Speed Distribution by Object Class",
                                labels={"class_name": "Object Class", "speed": "Speed (km/s)"})
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.header("Object Analysis")
            
            # Display object size vs depth
            st.subheader("Object Size vs. Depth")
            plot_object_size_vs_depth()
            
            # Additional object analysis
            if len(st.session_state.detection_data) > 0:
                df = pd.DataFrame(st.session_state.detection_data)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Display top detected classes
                    st.subheader("Top Detected Objects")
                    if "class_name" in df.columns:
                        top_classes = df["class_name"].value_counts().reset_index()
                        top_classes.columns = ["Object Class", "Count"]
                        st.dataframe(top_classes.head(10), use_container_width=True)
                
                with col2:
                    # Display average confidence by class
                    st.subheader("Average Confidence by Class")
                    if "class_name" in df.columns and "confidence" in df.columns:
                        conf_by_class = df.groupby("class_name")["confidence"].mean().reset_index()
                        conf_by_class.columns = ["Object Class", "Average Confidence"]
                        conf_by_class["Average Confidence"] = conf_by_class["Average Confidence"].map(lambda x: f"{x:.2f}")
                        st.dataframe(conf_by_class, use_container_width=True)
                
                # Create a 3D scatter plot (if enough data is available)
                if "depth" in df.columns and "speed" in df.columns and "confidence" in df.columns and len(df) > 5:
                    st.subheader("3D Visualization: Depth vs. Speed vs. Confidence")
                    fig = px.scatter_3d(df, x="depth", y="speed", z="confidence", color="class_name",
                                      title="3D Object Characteristics",
                                      labels={"depth": "Depth (m)", "speed": "Speed (km/s)", 
                                             "confidence": "Confidence", "class_name": "Object Class"})
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.header("Raw Data")
            
            # Display data table
            st.subheader("Detection Data")
            display_data_table()
            
            # Export options
            if len(st.session_state.detection_data) > 0:
                st.subheader("Export Options")
                
                df = pd.DataFrame(st.session_state.detection_data)
                
                # Download CSV button
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download as CSV",
                    data=csv_data,
                    file_name="stereo_detection_data.csv",
                    mime="text/csv"
                )
                
                # Download JSON button
                json_data = df.to_json(orient="records")
                st.download_button(
                    label="Download as JSON",
                    data=json_data,
                    file_name="stereo_detection_data.json",
                    mime="application/json"
                )
    else:
        # Display instructions when no data is available
        st.info("""
        ## How to use this dashboard:
        
        1. Select a model type and confidence threshold in the sidebar
        2. Click "Load Model" to initialize the YOLO model
        3. Choose between image or video analysis
        4. Select from sample data or upload your own stereo pairs
        5. Click "Analyze Images" or "Analyze Videos" to start processing
        6. Review the results in the various visualization tabs
        
        This dashboard provides comprehensive analysis of object detection, depth estimation, and speed calculation using stereo vision principles.
        """)
    
    # Display footer
    st.markdown("---")
    st.markdown("YOLO Stereo-Camera Visualization Dashboard | Developed with Streamlit, YOLOv5, and Plotly")

if __name__ == "__main__":
    main()