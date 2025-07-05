import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import pickle
import warnings

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

def process_webcam(model_path="./model/plank_7layer_dropout.pkl",
                  scaler_path="./model/input_scaler.pkl"):
    """Process webcam feed for plank detection in real-time"""
    
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

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Set desired window size (1920x1080 for full HD)
    window_width = 1920
    window_height = 1080
    
    # Create a named window and set it to full screen or large size
    cv2.namedWindow('Plank Form Analysis', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Plank Form Analysis', window_width, window_height)

    current_stage = ""
    prediction_probability_threshold = 0.6
    
    print("\nStarting webcam feed...")
    print("Press 'q' to quit")

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                break

            # Recolor image from BGR to RGB for mediapipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = pose.process(image_rgb)

            # Recolor back to BGR for OpenCV
            image_rgb.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            # Get the original frame dimensions
            original_height, original_width = image.shape[:2]
            
            # Make camera screen smaller by reducing the scale factor
            scale = 1.5  # adjust this value as needed
            
            # Calculate new dimensions
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            
            # Resize the image
            resized_image = cv2.resize(image, (new_width, new_height))
            
            # Create a black background
            background = np.zeros((window_height, window_width, 3), dtype=np.uint8)
            
            # Calculate position to center the image
            x_offset = (window_width - new_width) // 2
            y_offset = (window_height - new_height) // 2
            
            # Place the resized image on the background
            background[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_image
            
            # Draw heading panel (centered in the window)
            panel_width = 600
            panel_height = 60
            panel_x = (window_width - panel_width) // 2
            panel_y = 20
            
            cv2.rectangle(background, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1)
            cv2.rectangle(background, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (255, 255, 255), 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            # Center the text in the panel
            text = "PLANK ANALYSIS"
            text_size = cv2.getTextSize(text, font, 1, 2)[0]
            text_x = panel_x + (panel_width - text_size[0]) // 2
            text_y = panel_y + (panel_height + text_size[1]) // 2
            cv2.putText(background, text, (text_x, text_y), 
                       font, 1, (255, 255, 255), 2, cv2.LINE_AA)

            if not results.pose_landmarks:
                warning_panel_width = 600
                warning_panel_height = 50
                warning_panel_x = (window_width - warning_panel_width) // 2
                warning_panel_y = panel_y + panel_height + 30
                
                cv2.rectangle(background, (warning_panel_x, warning_panel_y), 
                             (warning_panel_x + warning_panel_width, warning_panel_y + warning_panel_height), 
                             (0, 0, 0), -1)
                cv2.rectangle(background, (warning_panel_x, warning_panel_y), 
                             (warning_panel_x + warning_panel_width, warning_panel_y + warning_panel_height), 
                             (0, 0, 255), 2)
                
                # Center the warning text
                warning_text = "NO HUMAN DETECTED"
                warning_text_size = cv2.getTextSize(warning_text, font, 0.8, 2)[0]
                warning_text_x = warning_panel_x + (warning_panel_width - warning_text_size[0]) // 2
                warning_text_y = warning_panel_y + (warning_panel_height + warning_text_size[1]) // 2
                cv2.putText(background, warning_text, (warning_text_x, warning_text_y), 
                           font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                # Draw landmarks on the original image before resizing for better quality
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(244, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1)
                )
                
                # Resize the image with landmarks
                resized_image = cv2.resize(image, (new_width, new_height))
                background[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_image

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
                    status_panel_x = (window_width - status_panel_width) // 2
                    status_panel_y = panel_y + panel_height + 30
                    
                    cv2.rectangle(background, (status_panel_x, status_panel_y), 
                                 (status_panel_x + status_panel_width, status_panel_y + status_panel_height), 
                                 (0, 0, 0), -1)
                    cv2.rectangle(background, (status_panel_x, status_panel_y), 
                                 (status_panel_x + status_panel_width, status_panel_y + status_panel_height), 
                                 (255, 255, 255), 2)
                    
                    # Center the status text
                    status_text = f"STATUS: {current_stage}"
                    status_text_size = cv2.getTextSize(status_text, font, 0.7, 2)[0]
                    status_text_x = status_panel_x + (status_panel_width - status_text_size[0]) // 2
                    status_text_y = status_panel_y + (status_panel_height + status_text_size[1]) // 2
                    cv2.putText(background, status_text, (status_text_x, status_text_y), 
                               font, 0.7, stage_color, 2, cv2.LINE_AA)

                except Exception as e:
                    error_panel_width = 600
                    error_panel_height = 50
                    error_panel_x = (window_width - error_panel_width) // 2
                    error_panel_y = panel_y + panel_height + 30
                    
                    cv2.rectangle(background, (error_panel_x, error_panel_y), 
                                 (error_panel_x + error_panel_width, error_panel_y + error_panel_height), 
                                 (0, 0, 0), -1)
                    cv2.rectangle(background, (error_panel_x, error_panel_y), 
                                 (error_panel_x + error_panel_width, error_panel_y + error_panel_height), 
                                 (0, 0, 255), 2)
                    
                    # Center the error text
                    error_text = f"Error: {str(e)[:40]}"
                    error_text_size = cv2.getTextSize(error_text, font, 0.6, 2)[0]
                    error_text_x = error_panel_x + (error_panel_width - error_text_size[0]) // 2
                    error_text_y = error_panel_y + (error_panel_height + error_text_size[1]) // 2
                    cv2.putText(background, error_text, (error_text_x, error_text_y),
                               font, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

            # Display the resulting frame
            cv2.imshow('Plank Form Analysis', background)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam processing stopped.")

if __name__ == "__main__":
    process_webcam()