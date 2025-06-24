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

        # Params for peak contraction error detection
        self.peak_contraction_angle = 1000
        self.peak_contraction_frame = None

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
        image = cv2.flip(image, 1)

        # Reduce size of a frame (optional, comment out if you want full resolution)
        # image = rescale_frame(image, 40)

        video_dimensions = [image.shape[1], image.shape[0]]

        # Recolor image from BGR to RGB for mediapipe
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        if not results.pose_landmarks:
            # Recolor back to BGR for display
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.putText(image, "No human detected", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("Bicep Curl Analysis", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Recolor image from RGB to BGR for OpenCV
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks and connections
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

            # Visualization - Centered metrics panel
            frame_width = image.shape[1]
            
            # Create centered black background for metrics
            panel_width = 700
            panel_height = 70
            panel_x = (frame_width - panel_width) // 2
            panel_y = 10
            
            # Draw black background rectangle
            cv2.rectangle(image, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1)
            
            # Draw border
            cv2.rectangle(image, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (255, 255, 255), 2)

            # Text settings
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            text_color = (255, 255, 255)  # White text
            thickness = 2
            
            # Row 1 - Labels
            y_offset = panel_y + 20
            cv2.putText(image, "RIGHT", (panel_x + 20, y_offset), font, font_scale, text_color, thickness)
            cv2.putText(image, "LEFT", (panel_x + 120, y_offset), font, font_scale, text_color, thickness)
            cv2.putText(image, "R_PC", (panel_x + 220, y_offset), font, font_scale, text_color, thickness)
            cv2.putText(image, "R_LUA", (panel_x + 300, y_offset), font, font_scale, text_color, thickness)
            cv2.putText(image, "L_PC", (panel_x + 400, y_offset), font, font_scale, text_color, thickness)
            cv2.putText(image, "L_LUA", (panel_x + 480, y_offset), font, font_scale, text_color, thickness)
            cv2.putText(image, "POSTURE", (panel_x + 580, y_offset), font, font_scale, text_color, thickness)

            # Row 2 - Values
            y_offset = panel_y + 50
            cv2.putText(image, str(right_arm_analysis.counter) if right_arm_analysis.is_visible else "UNK", 
                       (panel_x + 35, y_offset), font, font_scale, text_color, thickness)
            cv2.putText(image, str(left_arm_analysis.counter) if left_arm_analysis.is_visible else "UNK", 
                       (panel_x + 125, y_offset), font, font_scale, text_color, thickness)
            cv2.putText(image, str(right_arm_analysis.detected_errors["PEAK_CONTRACTION"]), 
                       (panel_x + 245, y_offset), font, font_scale, text_color, thickness)
            cv2.putText(image, str(right_arm_analysis.detected_errors["LOOSE_UPPER_ARM"]), 
                       (panel_x + 335, y_offset), font, font_scale, text_color, thickness)
            cv2.putText(image, str(left_arm_analysis.detected_errors["PEAK_CONTRACTION"]), 
                       (panel_x + 425, y_offset), font, font_scale, text_color, thickness)
            cv2.putText(image, str(left_arm_analysis.detected_errors["LOOSE_UPPER_ARM"]), 
                       (panel_x + 510, y_offset), font, font_scale, text_color, thickness)
            
            posture_text = f"{'C' if posture == 0 else 'L'} {prediction_probability:.2f}"
            cv2.putText(image, posture_text, (panel_x + 590, y_offset), font, 0.5, text_color, thickness)

            # * Visualize angles
            # Visualize LEFT arm calculated angles
            if left_arm_analysis.is_visible:
                cv2.putText(image, str(left_bicep_curl_angle), tuple(np.multiply(left_arm_analysis.elbow, video_dimensions).astype(int)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(image, str(left_ground_upper_arm_angle), tuple(np.multiply(left_arm_analysis.shoulder, video_dimensions).astype(int)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Visualize RIGHT arm calculated angles
            if right_arm_analysis.is_visible:
                cv2.putText(image, str(right_bicep_curl_angle), tuple(np.multiply(right_arm_analysis.elbow, video_dimensions).astype(int)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(right_ground_upper_arm_angle), tuple(np.multiply(right_arm_analysis.shoulder, video_dimensions).astype(int)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

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