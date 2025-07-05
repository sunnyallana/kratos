import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

# Drawing helpers
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Determine important landmarks for plank
IMPORTANT_LMS = [
    "NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "RIGHT_ELBOW", "LEFT_ELBOW",
    "RIGHT_WRIST", "LEFT_WRIST", "LEFT_HIP", "RIGHT_HIP",
]

# Generate all columns of the data frame
HEADERS = ["label"]  # Label column
for lm in IMPORTANT_LMS:
    HEADERS += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]

def rescale_frame(frame, percent=50):
    '''
    Rescale a frame from OpenCV to a certain percentage compare to its original frame
    '''
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

def save_frame_as_image(frame, message: str = None):
    '''
    Save a frame as image to display the error
    '''
    now = datetime.datetime.now()
    if message:
        cv2.putText(frame, message, (50, 150), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    print("Saving ...")
    cv2.imwrite(f"../data/logs/bicep_{now}.jpg", frame)

def calculate_angle(point1: list, point2: list, point3: list) -> float:
    '''
    Calculate the angle between 3 points
    Unit of the angle will be in Degree
    '''
    point1 = np.array(point1)
    point2 = np.array(point2)
    point3 = np.array(point3)

    # Calculate algo
    angleInRad = np.arctan2(point3[1] - point2[1], point3[0] - point2[0]) - np.arctan2(point1[1] - point2[1], point1[0] - point2[0])
    angleInDeg = np.abs(angleInRad * 180.0 / np.pi)
    angleInDeg = angleInDeg if angleInDeg <= 180 else 360 - angleInDeg
    return angleInDeg

def extract_important_keypoints(results, important_landmarks: list) -> list:
    '''
    Extract important keypoints from mediapipe pose detection
    '''
    landmarks = results.pose_landmarks.landmark
    data = []
    for lm in important_landmarks:
        keypoint = landmarks[mp_pose.PoseLandmark[lm].value]
        data.append([keypoint.x, keypoint.y, keypoint.z, keypoint.visibility])
    return np.array(data).flatten().tolist()

def draw_warning_box(image, text, x, y, width, height):
    '''
    Draw a prominent warning box with black background and red text
    '''
    # Create black background
    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 0, 0), -1)
    
    # Draw bright red border
    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 0, 255), 4)
    
    # Add warning text in bright red
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    text_color = (0, 0, 255)  # Bright red
    
    # Center text in box
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = x + (width - text_size[0]) // 2
    text_y = y + (height + text_size[1]) // 2
    
    cv2.putText(image, text, (text_x, text_y), font, font_scale, text_color, thickness)

class BicepPoseAnalysis:
    def __init__(self, side: str, stage_down_threshold: float, stage_up_threshold: float, peak_contraction_threshold: float, loose_upper_arm_angle_threshold: float, visibility_threshold: float):
        # Initialize thresholds
        self.stage_down_threshold = stage_down_threshold
        self.stage_up_threshold = stage_up_threshold
        self.peak_contraction_threshold = peak_contraction_threshold
        self.loose_upper_arm_angle_threshold = loose_upper_arm_angle_threshold
        self.visibility_threshold = visibility_threshold

        self.side = side
        self.counter = 0
        self.stage = "down"
        self.is_visible = True
        self.detected_errors = {
            "LOOSE_UPPER_ARM": 0,
            "PEAK_CONTRACTION": 0,
        }

        # Params for loose upper arm error detection
        self.loose_upper_arm = False
        self.current_loose_upper_arm = False  # Current frame error status

        # Params for peak contraction error detection
        self.peak_contraction_angle = 1000
        self.peak_contraction_frame = None
        self.current_peak_contraction_error = False  # Current frame error status

    def get_joints(self, landmarks) -> bool:
        '''
        Check for joints' visibility then get joints coordinate
        '''
        side = self.side.upper()

        # Check visibility
        joints_visibility = [
            landmarks[mp_pose.PoseLandmark[f"{side}_SHOULDER"].value].visibility,
            landmarks[mp_pose.PoseLandmark[f"{side}_ELBOW"].value].visibility,
            landmarks[mp_pose.PoseLandmark[f"{side}_WRIST"].value].visibility
        ]

        is_visible = all([vis > self.visibility_threshold for vis in joints_visibility])
        self.is_visible = is_visible

        if not is_visible:
            return self.is_visible

        # Get joints' coordinates
        self.shoulder = [
            landmarks[mp_pose.PoseLandmark[f"{side}_SHOULDER"].value].x,
            landmarks[mp_pose.PoseLandmark[f"{side}_SHOULDER"].value].y
        ]
        self.elbow = [
            landmarks[mp_pose.PoseLandmark[f"{side}_ELBOW"].value].x,
            landmarks[mp_pose.PoseLandmark[f"{side}_ELBOW"].value].y
        ]
        self.wrist = [
            landmarks[mp_pose.PoseLandmark[f"{side}_WRIST"].value].x,
            landmarks[mp_pose.PoseLandmark[f"{side}_WRIST"].value].y
        ]

        return self.is_visible

    def analyze_pose(self, landmarks, frame):
        '''
        - Bicep Counter
        - Errors Detection
        '''
        self.get_joints(landmarks)

        # Reset current frame error status
        self.current_loose_upper_arm = False
        self.current_peak_contraction_error = False

        # Cancel calculation if visibility is poor
        if not self.is_visible:
            return (None, None)

        # * Calculate curl angle for counter
        bicep_curl_angle = int(calculate_angle(self.shoulder, self.elbow, self.wrist))
        if bicep_curl_angle > self.stage_down_threshold:
            self.stage = "down"
        elif bicep_curl_angle < self.stage_up_threshold and self.stage == "down":
            self.stage = "up"
            self.counter += 1

        # * Calculate the angle between the upper arm (shoulder & joint) and the Y axis
        shoulder_projection = [self.shoulder[0], 1]  # Represent the projection of the shoulder to the X axis
        ground_upper_arm_angle = int(calculate_angle(self.elbow, self.shoulder, shoulder_projection))

        # * Evaluation for LOOSE UPPER ARM error
        if ground_upper_arm_angle > self.loose_upper_arm_angle_threshold:
            self.current_loose_upper_arm = True  # Set current frame error
            # Limit the saved frame
            if not self.loose_upper_arm:
                self.loose_upper_arm = True
                # save_frame_as_image(frame, f"Loose upper arm: {ground_upper_arm_angle}")
                self.detected_errors["LOOSE_UPPER_ARM"] += 1
        else:
            self.loose_upper_arm = False

        # * Evaluate PEAK CONTRACTION error
        if self.stage == "up" and bicep_curl_angle < self.peak_contraction_angle:
            # Save peaked contraction every rep
            self.peak_contraction_angle = bicep_curl_angle
            self.peak_contraction_frame = frame

        elif self.stage == "down":
            # * Evaluate if the peak is higher than the threshold if True, marked as an error then saved that frame
            if self.peak_contraction_angle != 1000 and self.peak_contraction_angle >= self.peak_contraction_threshold:
                self.current_peak_contraction_error = True  # Set current frame error
                # save_frame_as_image(self.peak_contraction_frame, f"{self.side} - Peak Contraction: {self.peak_contraction_angle}")
                self.detected_errors["PEAK_CONTRACTION"] += 1

            # Reset params
            self.peak_contraction_angle = 1000
            self.peak_contraction_frame = None

        return (bicep_curl_angle, ground_upper_arm_angle)

# Use camera instead of video file (0 is default camera, 1 for external camera)

cap = cv2.VideoCapture(0)

# Set camera resolution (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

VISIBILITY_THRESHOLD = 0.65

# Params for counter
STAGE_UP_THRESHOLD = 90
STAGE_DOWN_THRESHOLD = 120

# Params to catch FULL RANGE OF MOTION error
PEAK_CONTRACTION_THRESHOLD = 60

# LOOSE UPPER ARM error detection
LOOSE_UPPER_ARM = False
LOOSE_UPPER_ARM_ANGLE_THRESHOLD = 40

# STANDING POSTURE error detection
POSTURE_ERROR_THRESHOLD = 0.95
posture = 0

# Init analysis class
left_arm_analysis = BicepPoseAnalysis(
    side="left",
    stage_down_threshold=STAGE_DOWN_THRESHOLD,
    stage_up_threshold=STAGE_UP_THRESHOLD,
    peak_contraction_threshold=PEAK_CONTRACTION_THRESHOLD,
    loose_upper_arm_angle_threshold=LOOSE_UPPER_ARM_ANGLE_THRESHOLD,
    visibility_threshold=VISIBILITY_THRESHOLD
)

right_arm_analysis = BicepPoseAnalysis(
    side="right",
    stage_down_threshold=STAGE_DOWN_THRESHOLD,
    stage_up_threshold=STAGE_UP_THRESHOLD,
    peak_contraction_threshold=PEAK_CONTRACTION_THRESHOLD,
    loose_upper_arm_angle_threshold=LOOSE_UPPER_ARM_ANGLE_THRESHOLD,
    visibility_threshold=VISIBILITY_THRESHOLD
)

# Load input scaler
try:
    with open("./model/input_scaler.pkl", "rb") as f:
        input_scaler = pickle.load(f)
except FileNotFoundError:
    print("Warning: input_scaler.pkl not found. Skipping posture analysis.")
    input_scaler = None

# Load model
try:
    with open("./model/bicep_classifier_7layer.pkl", "rb") as f:
        DL_model = pickle.load(f)
except FileNotFoundError:
    print("Warning: bicep_dp.pkl not found. Skipping posture analysis.")
    DL_model = None

print("Starting camera capture... Press 'q' to quit")

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, image = cap.read()

        if not ret:
            print("Failed to capture image from camera")
            break

        # Flip image horizontally for mirror effect (optional)
        # image = cv2.flip(image, 1)

        video_dimensions = [image.shape[1], image.shape[0]]

        # Recolor image from BGR to RGB for mediapipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        results = pose.process(image_rgb)

        # Recolor image from RGB to BGR for OpenCV
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if not results.pose_landmarks:
            # Show no human detected on camera feed
            cv2.putText(image, "NO HUMAN DETECTED", (image.shape[1]//2 - 200, image.shape[0]//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.imshow("Bicep Curl Analysis", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Draw landmarks and connections on black background
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(244, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1)
        )

        # Make detection
        try:
            landmarks = results.pose_landmarks.landmark

            (left_bicep_curl_angle, left_ground_upper_arm_angle) = left_arm_analysis.analyze_pose(landmarks=landmarks, frame=image)
            (right_bicep_curl_angle, right_ground_upper_arm_angle) = right_arm_analysis.analyze_pose(landmarks=landmarks, frame=image)

            # Posture analysis (only if models are loaded)
            if input_scaler is not None and DL_model is not None:
                # Extract keypoints from frame for the input
                row = extract_important_keypoints(results, IMPORTANT_LMS)
                X = pd.DataFrame([row, ], columns=HEADERS[1:])
                X = pd.DataFrame(input_scaler.transform(X))

                # Make prediction and its probability
                prediction = DL_model.predict(X)
                predicted_class = np.argmax(prediction, axis=1)[0]
                prediction_probability = round(max(prediction.tolist()[0]), 2)

                if prediction_probability >= POSTURE_ERROR_THRESHOLD:
                    posture = predicted_class
            else:
                predicted_class = 0
                prediction_probability = 0.0

            # Enhanced Visualization - Keep camera feed with BLACK text backgrounds and PROMINENT RED warnings
            frame_width = image.shape[1]
            frame_height = image.shape[0]
            
            # Create header panel for rep counter with BLACK background
            panel_width = 400
            panel_height = 80
            panel_x = (frame_width - panel_width) // 2
            panel_y = 20
            
            # Draw black background with white border for counter
            cv2.rectangle(image, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1)
            cv2.rectangle(image, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (255, 255, 255), 2)

            # Text settings for counter
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            text_color = (255, 255, 255)  # White text
            thickness = 2
            
            # Title
            title_y = panel_y + 30
            cv2.putText(image, "BICEP CURLS", (panel_x + 120, title_y), font, font_scale, text_color, thickness)
            
            # Rep counters
            counter_y = panel_y + 65
            left_text = str(left_arm_analysis.counter) if left_arm_analysis.is_visible else "UNK"
            right_text = str(right_arm_analysis.counter) if right_arm_analysis.is_visible else "UNK"
            cv2.putText(image, f"LEFT: {left_text}", (panel_x + 30, counter_y), font, 0.8, text_color, thickness)
            cv2.putText(image, f"RIGHT: {right_text}", (panel_x + 220, counter_y), font, 0.8, text_color, thickness)

            # PROMINENT WARNING BOXES - Much larger and more visible
            warning_box_width = 400
            warning_box_height = 80
            warning_start_y = panel_y + panel_height + 40
            
            # Left arm warnings
            warning_y = warning_start_y
            if left_arm_analysis.current_loose_upper_arm:
                draw_warning_box(image, "LEFT ARM: LOOSE UPPER ARM!", 
                               50, warning_y, warning_box_width, warning_box_height)
                warning_y += warning_box_height + 20
                
            if left_arm_analysis.current_peak_contraction_error:
                draw_warning_box(image, "LEFT ARM: POOR CONTRACTION!", 
                               50, warning_y, warning_box_width, warning_box_height)
                warning_y += warning_box_height + 20

            # Right arm warnings
            warning_y = warning_start_y
            if right_arm_analysis.current_loose_upper_arm:
                draw_warning_box(image, "RIGHT ARM: LOOSE UPPER ARM!", 
                               frame_width - warning_box_width - 50, warning_y, 
                               warning_box_width, warning_box_height)
                warning_y += warning_box_height + 20
                
            if right_arm_analysis.current_peak_contraction_error:
                draw_warning_box(image, "RIGHT ARM: POOR CONTRACTION!", 
                               frame_width - warning_box_width - 50, warning_y, 
                               warning_box_width, warning_box_height)

            # Large warning text at bottom of screen with BLACK background
            bottom_warning_y = frame_height - 80
            active_warnings = []
            
            if left_arm_analysis.current_loose_upper_arm or right_arm_analysis.current_loose_upper_arm:
                active_warnings.append("KEEP UPPER ARMS STABLE!")
            if left_arm_analysis.current_peak_contraction_error or right_arm_analysis.current_peak_contraction_error:
                active_warnings.append("SQUEEZE HARDER AT THE TOP!")
                
            if active_warnings:
                for i, warning in enumerate(active_warnings):
                    # Calculate text size for background
                    font_scale = 1.2
                    thickness = 3
                    text_size = cv2.getTextSize(warning, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                    
                    # Draw black background for warning text
                    text_x = 50
                    text_y = bottom_warning_y + i * 50
                    cv2.rectangle(image, (text_x - 10, text_y - text_size[1] - 10), 
                                (text_x + text_size[0] + 10, text_y + 10), (0, 0, 0), -1)
                    
                    # Draw warning text in red
                    cv2.putText(image, warning, (text_x, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)

            # Visualize angles on joints
            if left_arm_analysis.is_visible:
                angle_color = (0, 0, 255) if left_arm_analysis.current_loose_upper_arm else (0, 255, 255)
                cv2.putText(image, str(left_bicep_curl_angle), 
                           tuple(np.multiply(left_arm_analysis.elbow, video_dimensions).astype(int)), 
                           cv2.FONT_HERSHEY_COMPLEX, 0.6, angle_color, 2, cv2.LINE_AA)
                cv2.putText(image, str(left_ground_upper_arm_angle), 
                           tuple(np.multiply(left_arm_analysis.shoulder, video_dimensions).astype(int)), 
                           cv2.FONT_HERSHEY_COMPLEX, 0.6, angle_color, 2, cv2.LINE_AA)

            if right_arm_analysis.is_visible:
                angle_color = (0, 0, 255) if right_arm_analysis.current_loose_upper_arm else (0, 255, 255)
                cv2.putText(image, str(right_bicep_curl_angle), 
                           tuple(np.multiply(right_arm_analysis.elbow, video_dimensions).astype(int)), 
                           cv2.FONT_HERSHEY_COMPLEX, 0.6, angle_color, 2, cv2.LINE_AA)
                cv2.putText(image, str(right_ground_upper_arm_angle), 
                           tuple(np.multiply(right_arm_analysis.shoulder, video_dimensions).astype(int)), 
                           cv2.FONT_HERSHEY_COMPLEX, 0.6, angle_color, 2, cv2.LINE_AA)

        except Exception as e:
            print(f"Error: {e}")

        cv2.imshow("Bicep Curl Analysis", image)

        # Press Q to close cv2 window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Fix bugs cannot close windows in MacOS
    for i in range(1, 5):
        cv2.waitKey(1)

print("Camera capture ended.")