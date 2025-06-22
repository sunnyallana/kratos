import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import pickle
import warnings
import time

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

def run_plank_detection_camera(model_path="./model/plank_7layer_dropout.pkl",
                              scaler_path="./model/input_scaler.pkl",
                              camera_index=0):
    """Run real-time plank detection using laptop camera"""

    print("Loading model and scaler...")
    try:
        # Load model and scaler
        with open(model_path, "rb") as f:
            deep_learning_model = pickle.load(f)

        with open(scaler_path, "rb") as f:
            input_scaler = pickle.load(f)

        print("Models loaded successfully!")
    except FileNotFoundError as e:
        print(f"Error loading models: {e}")
        print("Make sure you have trained the model and saved the scaler first!")
        return

    # Initialize camera
    print(f"Initializing camera (index: {camera_index})...")
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Error: Could not open camera. Try different camera index (0, 1, 2, etc.)")
        return

    # Set camera properties (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Camera initialized successfully!")
    print("Instructions:")
    print("- Position yourself in front of the camera")
    print("- Get into plank position")
    print("- Press 'q' to quit")
    print("- Press 's' to save current frame")

    current_stage = ""
    prediction_probability_threshold = 0.6
    frame_count = 0
    fps_counter = 0
    start_time = time.time()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                print("Failed to read from camera")
                break

            frame_count += 1
            fps_counter += 1

            # Calculate FPS every second
            current_time = time.time()
            if current_time - start_time >= 1.0:
                fps = fps_counter / (current_time - start_time)
                fps_counter = 0
                start_time = current_time
            else:
                fps = 0

            # Flip image horizontally for mirror effect
            image = cv2.flip(image, 1)

            # Reduce size of frame for better performance
            image = rescale_frame(image, 70)  # 70% of original size

            # Recolor image from BGR to RGB for mediapipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = pose.process(image_rgb)

            # Recolor back to BGR for OpenCV
            image_rgb.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            if not results.pose_landmarks:
                # Display message when no pose is detected
                cv2.putText(image, "NO POSE DETECTED", (50, 50),
                           cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(image, "Stand in front of camera", (50, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                # Draw landmarks and connections
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(244, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1)
                )

                # Make detection
                try:
                    # Extract keypoints from frame for the input
                    row = extract_important_keypoints(results)
                    X = pd.DataFrame([row], columns=HEADERS[1:])
                    X = pd.DataFrame(input_scaler.transform(X))

                    # Make prediction and its probability
                    prediction = deep_learning_model.predict(X, verbose=0)
                    predicted_class = np.argmax(prediction, axis=1)[0]
                    prediction_probability = max(prediction.tolist()[0])

                    # Evaluate model prediction
                    if predicted_class == 0 and prediction_probability >= prediction_probability_threshold:
                        current_stage = "Correct"
                        stage_color = (0, 255, 0)  # Green
                    elif predicted_class == 2 and prediction_probability >= prediction_probability_threshold:
                        current_stage = "Low back"
                        stage_color = (0, 0, 255)  # Red
                    elif predicted_class == 1 and prediction_probability >= prediction_probability_threshold:
                        current_stage = "High back"
                        stage_color = (0, 165, 255)  # Orange
                    else:
                        current_stage = "Unknown"
                        stage_color = (128, 128, 128)  # Gray

                    # Get image dimensions for positioning
                    h, w = image.shape[:2]

                    # Visualization - Centered metrics panel
                    frame_width = image.shape[1]
                    
                    # Create centered black background for metrics
                    panel_width = 600
                    panel_height = 80
                    panel_x = (frame_width - panel_width) // 2
                    panel_y = 10
                    
                    # Draw black background rectangle
                    cv2.rectangle(image, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1)
                    
                    # Draw border
                    cv2.rectangle(image, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (255, 255, 255), 2)

                    # Text settings
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale_title = 0.8
                    font_scale_text = 0.7
                    text_color = (255, 255, 255)  # White text
                    thickness = 2
                    
                    # Title
                    title_y = panel_y + 25
                    cv2.putText(image, "PLANK DETECTION", (panel_x + 20, title_y), font, font_scale_title, text_color, thickness)
                    
                    # Main status
                    status_y = panel_y + 55
                    cv2.putText(image, f"STATUS: {current_stage}", (panel_x + 20, status_y), font, font_scale_text, stage_color, thickness)
                    
                    # Class and probability on the right side of panel
                    cv2.putText(image, f"Class: {predicted_class}", (panel_x + 350, title_y), font, 0.6, text_color, 1)
                    cv2.putText(image, f"Confidence: {prediction_probability:.2f}", (panel_x + 350, status_y), font, 0.6, text_color, 1)
                    
                    # FPS on far right
                    if fps > 0:
                        cv2.putText(image, f"FPS: {fps:.1f}", (panel_x + 500, title_y), font, 0.6, text_color, 1)

                except Exception as e:
                    # Error display in centered panel
                    frame_width = image.shape[1]
                    panel_width = 600
                    panel_height = 60
                    panel_x = (frame_width - panel_width) // 2
                    panel_y = 10
                    
                    cv2.rectangle(image, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1)
                    cv2.rectangle(image, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (255, 255, 255), 2)
                    
                    cv2.putText(image, f"Error: {str(e)[:40]}", (panel_x + 20, panel_y + 35),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

            # Display instructions at bottom
            cv2.putText(image, "Press 'q' to quit, 's' to save frame",
                       (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Show the image
            cv2.imshow("Plank Detection - Real-time", image)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('s'):
                # Save current frame
                filename = f"plank_frame_{frame_count}.jpg"
                cv2.imwrite(filename, image)
                print(f"Frame saved as {filename}")

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

    # Fix for macOS window closing issue
    for i in range(1, 5):
        cv2.waitKey(1)

    print("Camera session ended.")

# Example usage
if __name__ == "__main__":
    print("Starting real-time plank detection...")

    # Try different camera indices if default doesn't work
    # Common indices: 0 (built-in), 1 (external USB camera)
    run_plank_detection_camera(camera_index=0)