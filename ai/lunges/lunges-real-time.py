import cv2
import pickle
import mediapipe as mp
import numpy as np
import pandas as pd
from pathlib import Path

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    """Calculate the angle between three points"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def extract_important_keypoints(results, important_landmarks):
    """Extract important landmarks from MediaPipe results"""
    pose = mp_pose.PoseLandmark
    row = []
    
    for landmark in important_landmarks:
        landmark = pose[landmark].value
        landmark = results.pose_landmarks.landmark[landmark]
        row.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
    
    return row

class LungeDetection:
    STAGE_ML_MODEL_PATH = "model/lunge_stage_model.pkl"
    ERR_ML_MODEL_PATH = "model/lunge_err_model.pkl"
    INPUT_SCALER_PATH = "model/lunge_input_scaler.pkl"

    PREDICTION_PROB_THRESHOLD = 0.8
    KNEE_ANGLE_THRESHOLD = [60, 125]

    def __init__(self) -> None:
        self.init_important_landmarks()
        self.load_machine_learning_model()

        self.current_stage = ""
        self.counter = 0
        self.results = []
        self.has_error = False
        self.right_knee_pos = None
        self.left_knee_pos = None

    def init_important_landmarks(self) -> None:
        """Determine Important landmarks for lunge detection"""
        self.important_landmarks = [
            "NOSE",
            "LEFT_SHOULDER",
            "RIGHT_SHOULDER",
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
        self.headers = ["label"]  # Label column

        for lm in self.important_landmarks:
            self.headers += [
                f"{lm.lower()}_x",
                f"{lm.lower()}_y",
                f"{lm.lower()}_z",
                f"{lm.lower()}_v",
            ]

    def load_machine_learning_model(self) -> None:
        """Load machine learning model"""
        if (not Path(self.STAGE_ML_MODEL_PATH).exists() or 
            not Path(self.INPUT_SCALER_PATH).exists() or 
            not Path(self.ERR_ML_MODEL_PATH).exists()):
            raise Exception("Cannot find lunge model files for prediction")

        try:
            with open(self.ERR_ML_MODEL_PATH, "rb") as f:
                self.err_model = pickle.load(f)

            with open(self.STAGE_ML_MODEL_PATH, "rb") as f:
                self.stage_model = pickle.load(f)

            with open(self.INPUT_SCALER_PATH, "rb") as f2:
                self.input_scaler = pickle.load(f2)
        except Exception as e:
            raise Exception(f"Error loading model, {e}")

    def analyze_knee_angle(self, mp_results, stage: str, angle_thresholds: list, knee_over_toe: bool = False):
        """Calculate angle of each knee while performer at the DOWN position"""
        results = {
            "error": None,
            "right": {"error": None, "angle": None},
            "left": {"error": None, "angle": None},
        }

        landmarks = mp_results.pose_landmarks.landmark

        # Calculate right knee angle and position
        right_hip = [
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
        ]
        self.right_knee_pos = [
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
        ]
        right_ankle = [
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
        ]
        results["right"]["angle"] = calculate_angle(right_hip, self.right_knee_pos, right_ankle)

        # Calculate left knee angle and position
        left_hip = [
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
        ]
        self.left_knee_pos = [
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
        ]
        left_ankle = [
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
        ]
        results["left"]["angle"] = calculate_angle(left_hip, self.left_knee_pos, left_ankle)

        if stage != "down":
            return results

        # Ignore checking for knee angle error if knee_over_toe error occurs
        if knee_over_toe:
            return results

        # Evaluation
        results["error"] = False

        if angle_thresholds[0] <= results["right"]["angle"] <= angle_thresholds[1]:
            results["right"]["error"] = False
        else:
            results["right"]["error"] = True
            results["error"] = True

        if angle_thresholds[0] <= results["left"]["angle"] <= angle_thresholds[1]:
            results["left"]["error"] = False
        else:
            results["left"]["error"] = True
            results["error"] = True

        return results

    def detect(self, mp_results, image):
        """Make Lunge Errors detection"""
        try:
            video_dimensions = [image.shape[1], image.shape[0]]

            # Model prediction
            row = extract_important_keypoints(mp_results, self.important_landmarks)
            X = pd.DataFrame([row], columns=self.headers[1:])
            X = pd.DataFrame(self.input_scaler.transform(X))

            # Stage prediction
            stage_predicted_class = self.stage_model.predict(X)[0]
            stage_prediction_prob = self.stage_model.predict_proba(X)[0].max()

            # Update stage and counter
            if stage_prediction_prob >= self.PREDICTION_PROB_THRESHOLD:
                if stage_predicted_class == "D" and self.current_stage in ["init", "mid"]:
                    self.counter += 1
                self.current_stage = {"I": "init", "M": "mid", "D": "down"}[stage_predicted_class]

            # Error detection (only in 'down' stage)
            self.has_error = False
            if self.current_stage == "down":
                err_predicted_class = self.err_model.predict(X)[0]
                err_prediction_prob = self.err_model.predict_proba(X)[0].max()
                
                if err_prediction_prob >= self.PREDICTION_PROB_THRESHOLD:
                    self.has_error = (err_predicted_class == "L")

            # Knee angle analysis
            analyzed_results = self.analyze_knee_angle(
                mp_results=mp_results,
                stage=self.current_stage,
                angle_thresholds=self.KNEE_ANGLE_THRESHOLD,
                knee_over_toe=self.has_error
            )

            # Update error status
            if self.current_stage == "down":
                self.has_error = self.has_error or analyzed_results["error"]

            # Visualization
            # 1. Draw pose landmarks
            mp_drawing.draw_landmarks(
                image,
                mp_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255) if self.has_error else (0, 255, 0)), 
                mp_drawing.DrawingSpec(color=(0, 0, 255) if self.has_error else (0, 255, 0))
            )
            
            # 2. Create black translucent overlay
            overlay = image.copy()
            cv2.rectangle(overlay, (0, 0), (image.shape[1], 60), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
            
            # 3. Display metrics (WHITE TEXT)
            cv2.putText(
                image,
                f"REPS: {self.counter} | STATUS: {'CORRECT' if not self.has_error else 'INCORRECT'}",
                (image.shape[1]//2 - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),  # WHITE
                2,
                cv2.LINE_AA,
            )

            # 4. Display knee angles
            if self.right_knee_pos and analyzed_results["right"]["angle"]:
                cv2.putText(
                    image,
                    str(int(analyzed_results["right"]["angle"])),
                    tuple(np.multiply(self.right_knee_pos, video_dimensions).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                
            if self.left_knee_pos and analyzed_results["left"]["angle"]:
                cv2.putText(
                    image,
                    str(int(analyzed_results["left"]["angle"])),
                    tuple(np.multiply(self.left_knee_pos, video_dimensions).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        except Exception as e:
            print(f"Detection error: {e}")

def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Initialize Pose detection
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Initialize Lunge detector
    detector = LungeDetection()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Detect lunges if pose found
        if results.pose_landmarks:
            detector.detect(results, image)
        
        # Display
        cv2.imshow('Lunge Trainer', image)
        
        # Exit on 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print(f"Final rep count: {detector.counter}")

if __name__ == "__main__":
    main()