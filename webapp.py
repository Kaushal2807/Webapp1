import glob
import streamlit as st
import torch
import cv2
import os
import pathlib
from PIL import Image, ImageDraw, ImageFont
import time
import math
import numpy as np

# FIX: Don't modify the pathlib.Path class, just use the appropriate Path for the OS
# REMOVED: pathlib.Path = pathlib.WindowsPath if os.name == 'nt' else pathlib.PosixPath

st.set_page_config(layout="wide")

cfg_model_path = 'models/best.pt'
model = None
confidence = 0.25

def download_data():
    # Get the Google Drive URL from the environment variable
    drive_url = os.getenv("DATA_DRIVE_URL")  # Updated to match .env file
    print("drive_url", drive_url)
    
    if not drive_url:
        raise ValueError("DATA_DRIVE_URL is not set in the environment variables.")
    
    # Check if data directory already exists
    if not os.path.exists("data"):
        os.makedirs("data", exist_ok=True)
        
        import gdown
        import sys
        
        try:
            print("Downloading folder from Google Drive...")
            # Extract folder ID if URL is in standard format
            folder_id = None
            
            # Handle various URL formats
            if "folders/" in drive_url:
                folder_id = drive_url.split("folders/")[1].split("/")[0].split("?")[0]
            elif "id=" in drive_url:
                folder_id = drive_url.split("id=")[1].split("&")[0]
            elif "drive.google.com" in drive_url and "/d/" in drive_url:
                folder_id = drive_url.split("/d/")[1].split("/")[0]
            else:
                # Assume the URL is the ID itself
                folder_id = drive_url
                
            print(f"Detected folder ID: {folder_id}")
            
            # Force reinstall gdown to ensure latest version
            print("Ensuring latest version of gdown is installed...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "gdown"])
            
            # Try using various download methods
            try:
                print("Attempting to download using folder ID method...")
                gdown.download_folder(id=folder_id, output="data", quiet=False, remaining_ok=True)
            except Exception as e1:
                print(f"First download attempt failed: {str(e1)}")
                try:
                    print("Attempting to download using URL method...")
                    gdown.download_folder(url=drive_url, output="data", quiet=False, remaining_ok=True)
                except Exception as e2:
                    print(f"Second download attempt failed: {str(e2)}")
                    print("Attempting final download method with direct URL...")
                    direct_url = f"https://drive.google.com/drive/folders/{folder_id}"
                    gdown.download_folder(url=direct_url, output="data", quiet=False, remaining_ok=True)
                
            print("Folder download complete!")
            
            # Verify download success
            if len(os.listdir("data")) == 0:
                print("Warning: Data folder is empty after download. This might indicate an issue with permissions.")
                print("Please ensure your Google Drive folder has been shared with 'Anyone with the link'")
        except Exception as e:
            print(f"Error downloading folder: {str(e)}")
            print("Please ensure your Google Drive URL is correct and the folder has public access permissions.")
            print("You may need to run: pip install --upgrade gdown")

def load_model(device, model_type='yolov5s'):
    # Fix for torch.classes path issue
    try:
        model_ = torch.hub.load('ultralytics/yolov5', model_type, pretrained=True, force_reload=True)
        model_.to(device)
        print(f"Pretrained Model '{model_type}' loaded on {device}")
        return model_
    except Exception as e:
        print(f"Error loading model: {e}")
        # Alternative loading method
        model_ = torch.hub.load('ultralytics/yolov5', model_type, pretrained=True, trust_repo=True)
        model_.to(device)
        print(f"Pretrained Model '{model_type}' loaded using alternative method on {device}")
        return model_

f = 700  # Focal length of the camera (in pixels)
B = 0.1  # Baseline distance between the two cameras (in meters)

def calculate_depth(x1, x2):
    """Calculate the depth Z of the object using stereo vision."""
    if x1 == x2:
        return 1000.0  # Return a large number instead of raising an error
    Z = (f * B) / abs(x1 - x2)  # Use absolute difference to avoid negative values
    return Z

def calculate_speed(position1, position2, time_interval):
    """
    Calculate the speed of the object.
    position1 and position2 are tuples (x1, y1, x2, y2, t),
    where t is the timestamp of the position.
    time_interval is the difference in time between the two positions.
    """
    x1_1, y1_1, x2_1, y2_1, t1 = position1
    x1_2, y1_2, x2_2, y2_2, t2 = position2
    
    # Prevent division by zero or very small time intervals
    if time_interval <= 0.001:
        time_interval = 0.001
        
    # Calculate depth for both positions
    try:
        Z1 = calculate_depth(x1_1, x2_1)
        Z2 = calculate_depth(x1_2, x2_2)
        # Calculate distance moved in 3D space
        distance = math.sqrt((x1_2 - x1_1)**2 + (y1_2 - y1_1)**2 + (Z2 - Z1)**2)
        # Calculate speed
        speed = distance / time_interval
        return speed
    except Exception as e:
        print(f"Error calculating speed: {e}")
        return 0.0

def image_input(data_src):
    img_fileL, img_fileR = None, None
    fns = 40

    if data_src == 'Sample':
        img_files_L = glob.glob('data/sample_images/left/*')
        img_files_R = glob.glob('data/sample_images/right/*')
        if img_files_L and img_files_R:
            img_fileL = img_files_L[0]
            img_fileR = img_files_R[0]
    else:
        img_bytesL = st.sidebar.file_uploader("Upload a left image", type=['png', 'jpeg', 'jpg'])
        if img_bytesL:
            os.makedirs("data/uploaded_data", exist_ok=True)
            img_fileL = f"data/uploaded_data/uploadL.{img_bytesL.name.split('.')[-1]}"
            Image.open(img_bytesL).save(img_fileL)

        img_bytesR = st.sidebar.file_uploader("Upload a right image", type=['png', 'jpeg', 'jpg'])
        if img_bytesR:
            os.makedirs("data/uploaded_data", exist_ok=True)
            img_fileR = f"data/uploaded_data/uploadR.{img_bytesR.name.split('.')[-1]}"
            Image.open(img_bytesR).save(img_fileR)

    if img_fileL and img_fileR:
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_fileL, caption="Left Image")
        with col2:
            st.image(img_fileR, caption="Right Image")
        
        col3, col4 = st.columns(2)
        with col3:
            imgL, _ = infer_image(img_fileL, fns)
            if imgL is not None:
                st.image(imgL, caption="Left Image prediction")
        with col4:
            imgR, _ = infer_image(img_fileR, fns)
            if imgR is not None:
                st.image(imgR, caption="Right Image prediction")

def video_input(data_src):
    vid_fileL = None
    vid_fileR = None
    fns = 20  # Default font size for video frames
    
    if data_src == 'Sample':
        vid_fileL = "data/sample_videos/left/sampleL.mp4"
        vid_fileR = "data/sample_videos/right/sampleR.mp4"
    else:
        vid_bytesL = st.sidebar.file_uploader("Upload a left video", type=['mp4', 'mpv', 'avi'])
        if vid_bytesL:
            os.makedirs("data/uploaded_data", exist_ok=True)
            vid_fileL = f"data/uploaded_data/uploadL.{vid_bytesL.name.split('.')[-1]}"
            with open(vid_fileL, 'wb') as out:
                out.write(vid_bytesL.read())
        
        vid_bytesR = st.sidebar.file_uploader("Upload a Right video", type=['mp4', 'mpv', 'avi'])
        if vid_bytesR:
            os.makedirs("data/uploaded_data", exist_ok=True)
            vid_fileR = f"data/uploaded_data/uploadR.{vid_bytesR.name.split('.')[-1]}"
            with open(vid_fileR, 'wb') as out:
                out.write(vid_bytesR.read())

    if vid_fileL and vid_fileR:
        if not os.path.exists(vid_fileL) or not os.path.exists(vid_fileR):
            st.error("Video files not found. Please check the paths.")
            return
            
        capL = cv2.VideoCapture(vid_fileL)
        capR = cv2.VideoCapture(vid_fileR)
        
        if not capL.isOpened() or not capR.isOpened():
            st.error("Error opening video files. Please check the format.")
            return
            
        custom_size = st.sidebar.checkbox("Custom frame size")
        width = 640
        height = 480
        if custom_size:
            width = st.sidebar.number_input("Width", min_value=120, step=20, value=width)
            height = st.sidebar.number_input("Height", min_value=120, step=20, value=height)
        fps = 0
        st1, st2, st3 = st.columns(3)
        with st1:
            st.markdown("## Height")
            st1_text = st.markdown(f"{height}")
        with st2:
            st.markdown("## Width")
            st2_text = st.markdown(f"{width}")
        with st3:
            st.markdown("## FPS")
            st3_text = st.markdown(f"{fps}")
        
        # Create two separate placeholders for left and right frames
        col_left, col_right = st.columns(2)
        with col_left:
            left_frame_placeholder = st.empty()
        with col_right:
            right_frame_placeholder = st.empty()
        
        prev_time = 0
        curr_time = 0
        positionl = []
        positionr = []
        frn = 0
        tm = []
        
        # Try to load font, fallback to default if not available
        try:
            font = ImageFont.truetype("arialbd.ttf", 20)
        except IOError:
            try:
                font = ImageFont.truetype("Arial Bold.ttf", 20)
            except IOError:
                font = ImageFont.load_default()
        
        while True:
            retL, frameL = capL.read()
            retR, frameR = capR.read()
            
            if not retL or not retR:
                st.write("Can't read frame, stream ended? Exiting ....")
                break
                
            frameL = cv2.resize(frameL, (width, height))
            frameL = cv2.cvtColor(frameL, cv2.COLOR_BGR2RGB)
            output_imgL, rrl = infer_image(frameL, fns)
            
            frameR = cv2.resize(frameR, (width, height))
            frameR = cv2.cvtColor(frameR, cv2.COLOR_BGR2RGB)
            output_imgR, rrr = infer_image(frameR, fns)
            
            if output_imgL is None or output_imgR is None:
                continue
                
            positionl.append(rrl)
            positionr.append(rrr)
            current_time = time.time()
            tm.append(current_time)
            
            if len(positionl) > 2 and rrl is not None and len(rrl.pred) > 0 and len(rrl.pred[0]) > 0:
                time2 = tm[len(tm)-1]
                time1 = tm[len(tm)-2]
                time_interval = time2 - time1
                
                for idx, det in enumerate(rrl.pred[0]):
                    if len(det) == 0:
                        continue
                    
                    x1, y1, x2, y2 = map(float, det[:4].cpu().numpy())
                    
                    # Safely calculate speed
                    try:
                        speed = calculate_speed(
                            [x1, y1, x1+25, y1+25, time1], 
                            [x1+50, y1+50, x1+75, y1+75, time2], 
                            time_interval
                        ) * 0.8
                        
                        # Create a draw object if the output_imgL is a PIL Image
                        if isinstance(output_imgL, Image.Image):
                            draw = ImageDraw.Draw(output_imgL)
                            speed_text = f"{speed:.2f}km/hr"
                            draw.text((int(x1), int(y1)), speed_text, fill="yellow", font=font)
                    except Exception as e:
                        print(f"Error calculating or displaying speed for left image: {e}")
            
            if len(positionr) > 2 and rrr is not None and len(rrr.pred) > 0 and len(rrr.pred[0]) > 0:
                time2 = tm[len(tm)-1]
                time1 = tm[len(tm)-2]
                time_interval = time2 - time1
                
                for idx, det in enumerate(rrr.pred[0]):
                    if len(det) == 0:
                        continue
                    
                    x1, y1, x2, y2 = map(float, det[:4].cpu().numpy())
                    
                    # Safely calculate speed
                    try:
                        speed = calculate_speed(
                            [x1, y1, x1+25, y1+25, time1], 
                            [x1+50, y1+50, x1+75, y1+75, time2], 
                            time_interval
                        ) * 0.8
                        
                        # Create a draw object if the output_imgR is a PIL Image
                        if isinstance(output_imgR, Image.Image):
                            draw = ImageDraw.Draw(output_imgR)
                            speed_text = f"{speed:.2f}km/hr"
                            draw.text((int(x1), int(y1)), speed_text, fill="yellow", font=font)
                    except Exception as e:
                        print(f"Error calculating or displaying speed for right image: {e}")
            
            # Display left and right frames separately
            if isinstance(output_imgL, Image.Image):
                left_frame_placeholder.image(output_imgL, caption="Left Frame", use_column_width=True)
            
            if isinstance(output_imgR, Image.Image):
                right_frame_placeholder.image(output_imgR, caption="Right Frame", use_column_width=True)
            
            curr_time = time.time()
            fps = 1 / max(curr_time - prev_time, 0.001)  # Prevent division by zero
            prev_time = curr_time
            
            st1_text.markdown(f"**{height}**")
            st2_text.markdown(f"**{width}**")
            st3_text.markdown(f"**{fps:.2f}**")
            
            frn = frn + 1
        
        capL.release()
        capR.release()

def infer_image(img, fns, size=None):
    global model
    if model is None:
        st.error("Model is not loaded. Please select a valid model.")
        return None, None
    
    model.conf = confidence  # Set confidence threshold
    
    # Handle different input types (file path or numpy array)
    if isinstance(img, str):
        if not os.path.exists(img):
            st.error(f"Image file not found: {img}")
            return None, None
    
    try:
        result = model(img, size=size) if size else model(img)
        result.render()
        
        # Convert the result image to PIL Image
        if len(result.ims) > 0:
            image = Image.fromarray(result.ims[0])
            draw = ImageDraw.Draw(image)
            
            try:
                font = ImageFont.truetype("arial.ttf", fns)
            except IOError:
                try:
                    font = ImageFont.truetype("Arial.ttf", fns)
                except IOError:
                    font = ImageFont.load_default()
            
            if len(result.pred) > 0 and len(result.pred[0]) > 0:
                for idx, det in enumerate(result.pred[0]):
                    if len(det) == 0:
                        continue
                    
                    x1, y1, x2, y2 = map(float, det[:4].cpu().numpy())
                    label = f"X: {x1:.2f}, Y: {y1:.2f}"
                    # draw.text((int(x1), int(y1)), label, fill="white", font=font)
            
            return image, result
        else:
            return None, None
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, None

def main():
    download_data()
    global model, confidence, cfg_model_path
    st.title("YOLO Stereo-Camera Object Detection and Speed Estimation")
    
    if not os.path.isfile(cfg_model_path) and not st.session_state.get('model_warning_shown', False):
        st.warning("Model file not found at expected path. Using pre-trained models instead.", icon="⚠️")
        st.session_state['model_warning_shown'] = True
    
    model_type = st.sidebar.selectbox("Select Model Type", ['yolov5s', 'yolov5m', 'yolov5l', 'yolov5x'])
    
    # Load model with error handling
    try:
        model = load_model("cpu", model_type=model_type)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please check your internet connection and try again.")
        return
    
    confidence = st.sidebar.slider('Confidence', min_value=0.1, max_value=1.0, value=0.45)
    
    # Class selection with error handling
    try:
        if st.sidebar.checkbox("Select Classes"):
            model_names = list(model.names.values())
            assigned_class = st.sidebar.multiselect("Select Classes", model_names, default=[model_names[0]])
            classes = [model_names.index(name) for name in assigned_class]
            model.classes = classes
        else:
            model.classes = list(model.names.keys())
    except Exception as e:
        st.error(f"Error setting up class filter: {e}")
    
    st.sidebar.markdown("---")
    input_option = st.sidebar.radio("Select type: ", ['image', 'video'])
    data_src = st.sidebar.radio("Select input source: ", ['Sample', 'Own data'])
    
    # Create necessary directories
    os.makedirs("data/sample_images/left", exist_ok=True)
    os.makedirs("data/sample_images/right", exist_ok=True)
    os.makedirs("data/sample_videos/left", exist_ok=True)
    os.makedirs("data/sample_videos/right", exist_ok=True)
    os.makedirs("data/uploaded_data", exist_ok=True)
    
    if input_option == 'image':
        image_input(data_src)
    else:
        video_input(data_src)

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")