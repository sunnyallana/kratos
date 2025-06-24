import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import pickle
import warnings
import time
import os
import sys

warnings.filterwarnings('ignore')

# Drawing helpers
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Determine important landmarks for plank
IMPORTANT_LMS = [
    "NOSE",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
    "LEFT_HEEL",
    "RIGHT_HEEL",
    "LEFT_FOOT_INDEX",
    "RIGHT_FOOT_INDEX",
]

# Generate all columns of the data frame
HEADERS = ["label"]  # Label column
for lm in IMPORTANT_LMS:
    HEADERS += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]

def extract_important_keypoints(results) -> list:
    """Extract important keypoints from mediapipe pose detection"""
    landmarks = results.pose_landmarks.landmark
    data = []
    for lm in IMPORTANT_LMS:
        keypoint = landmarks[mp_pose.PoseLandmark[lm].value]
        data.append([keypoint.x, keypoint.y, keypoint.z, keypoint.visibility])
    return np.array(data).flatten().tolist()

def rescale_frame(frame, percent=50):
    """Rescale a frame to a certain percentage compare to its original frame"""
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
    """Call in a loop to create terminal progress bar"""
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    if iteration == total:
        print()

def process_video(input_video_path, output_video_path, 
                 model_path="./model/plank_7layer_dropout.pkl",
                 scaler_path="./model/input_scaler.pkl"):
    """Process video file for plank detection and save output"""
    
    print(f"Processing video: {input_video_path}")
    print(f"Output will be saved to: {output_video_path}")
    
    # Load model and scaler
    print("Loading model and scaler...")
    try:
        with open(model_path, "rb") as f:
            deep_learning_model = pickle.load(f)

        with open(scaler_path, "rb") as f:
            input_scaler = pickle.load(f)
        print("Models loaded successfully!")
    except FileNotFoundError as e:
        print(f"Error loading models: {e}")
        print("Make sure you have trained the model and saved the scaler first!")
        return

    # Initialize video capture
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_video_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    current_stage = ""
    prediction_probability_threshold = 0.6
    frame_count = 0
    start_time = time.time()
    
    print(f"\nTotal frames to process: {total_frames}")
    print("Starting processing...")

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                break

            frame_count += 1
            
            # Print progress every 10 frames or on last frame
            if frame_count % 10 == 0 or frame_count == total_frames:
                print_progress_bar(frame_count, total_frames, prefix='Progress:', suffix=f'Frame {frame_count}/{total_frames}', length=40)

            # Recolor image from BGR to RGB for mediapipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = pose.process(image_rgb)

            # Recolor back to BGR for OpenCV
            image_rgb.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            # Draw heading panel
            panel_width = 600
            panel_height = 50
            panel_x = (frame_width - panel_width) // 2
            panel_y = 10
            
            cv2.rectangle(image, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1)
            cv2.rectangle(image, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (255, 255, 255), 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, "PLANK ANALYSIS", (panel_x + 180, panel_y + 35), 
                       font, 1, (255, 255, 255), 2, cv2.LINE_AA)

            if not results.pose_landmarks:
                warning_panel_width = 600
                warning_panel_height = 50
                warning_panel_x = (frame_width - warning_panel_width) // 2
                warning_panel_y = panel_y + panel_height + 20
                
                cv2.rectangle(image, (warning_panel_x, warning_panel_y), 
                             (warning_panel_x + warning_panel_width, warning_panel_y + warning_panel_height), 
                             (0, 0, 0), -1)
                cv2.rectangle(image, (warning_panel_x, warning_panel_y), 
                             (warning_panel_x + warning_panel_width, warning_panel_y + warning_panel_height), 
                             (0, 0, 255), 2)
                
                cv2.putText(image, "NO HUMAN DETECTED", (warning_panel_x + 200, warning_panel_y + 35), 
                           font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(244, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1)
                )

                try:
                    row = extract_important_keypoints(results)
                    X = pd.DataFrame([row], columns=HEADERS[1:])
                    X = pd.DataFrame(input_scaler.transform(X))

                    prediction = deep_learning_model.predict(X, verbose=0)
                    predicted_class = np.argmax(prediction, axis=1)[0]
                    prediction_probability = max(prediction.tolist()[0])

                    if predicted_class == 0 and prediction_probability >= prediction_probability_threshold:
                        current_stage = "CORRECT PLANK FORM"
                        stage_color = (0, 255, 0)
                    elif predicted_class == 2 and prediction_probability >= prediction_probability_threshold:
                        current_stage = "LOW BACK DETECTED"
                        stage_color = (0, 0, 255)
                    elif predicted_class == 1 and prediction_probability >= prediction_probability_threshold:
                        current_stage = "HIGH BACK DETECTED"
                        stage_color = (0, 0, 255)
                    else:
                        current_stage = "UNKNOWN POSE"
                        stage_color = (128, 128, 128)

                    status_panel_width = 600
                    status_panel_height = 50
                    status_panel_x = (frame_width - status_panel_width) // 2
                    status_panel_y = panel_y + panel_height + 20
                    
                    cv2.rectangle(image, (status_panel_x, status_panel_y), 
                                 (status_panel_x + status_panel_width, status_panel_y + status_panel_height), 
                                 (0, 0, 0), -1)
                    cv2.rectangle(image, (status_panel_x, status_panel_y), 
                                 (status_panel_x + status_panel_width, status_panel_y + status_panel_height), 
                                 (255, 255, 255), 2)
                    
                    cv2.putText(image, f"STATUS: {current_stage}", (status_panel_x + 20, status_panel_y + 35), 
                               font, 0.7, stage_color, 2, cv2.LINE_AA)

                except Exception as e:
                    error_panel_width = 600
                    error_panel_height = 50
                    error_panel_x = (frame_width - error_panel_width) // 2
                    error_panel_y = panel_y + panel_height + 20
                    
                    cv2.rectangle(image, (error_panel_x, error_panel_y), 
                                 (error_panel_x + error_panel_width, error_panel_y + error_panel_height), 
                                 (0, 0, 0), -1)
                    cv2.rectangle(image, (error_panel_x, error_panel_y), 
                                 (error_panel_x + error_panel_width, error_panel_y + error_panel_height), 
                                 (0, 0, 255), 2)
                    
                    cv2.putText(image, f"Error: {str(e)[:40]}", (error_panel_x + 20, error_panel_y + 35),
                               font, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.putText(image, f"Processing frame: {frame_count}", 
                       (10, frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            out.write(image)

    # Calculate processing time
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"\nProcessed {frame_count} frames in {processing_time:.2f} seconds")
    print(f"Average FPS: {frame_count/processing_time:.2f}")

    # Clean up
    cap.release()
    out.release()
    print(f"Video processing complete. Output saved to {output_video_path}")

if __name__ == "__main__":
    input_video = "plank-video.mp4"
    output_video = "output_plank_analysis.mp4"
    
    if not os.path.exists(input_video):
        print(f"Error: Input video file '{input_video}' not found!")
    else:
        process_video(input_video, output_video)