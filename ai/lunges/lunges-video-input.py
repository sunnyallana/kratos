import cv2
import pickle
import mediapipe as mp
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import time

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

    def clear_results(self) -> None:
        self.results = []
        self.counter = 0
        self.current_stage = ""
        self.has_error = False

    def detect(self, mp_results, image, timestamp) -> None:
        """Make Lunge Errors detection"""
        try:
            video_dimensions = [image.shape[1], image.shape[0]]

            # Model prediction for LUNGE counter
            row = extract_important_keypoints(mp_results, self.important_landmarks)
            X = pd.DataFrame([row], columns=self.headers[1:])
            X = pd.DataFrame(self.input_scaler.transform(X))

            # Make prediction and its probability
            stage_predicted_class = self.stage_model.predict(X)[0]
            stage_prediction_probabilities = self.stage_model.predict_proba(X)[0]
            stage_prediction_probability = round(
                stage_prediction_probabilities[stage_prediction_probabilities.argmax()],
                2,
            )

            # Evaluate stage prediction for counter
            if (
                stage_predicted_class == "I"
                and stage_prediction_probability >= self.PREDICTION_PROB_THRESHOLD
            ):
                self.current_stage = "init"
            elif (
                stage_predicted_class == "M"
                and stage_prediction_probability >= self.PREDICTION_PROB_THRESHOLD
            ):
                self.current_stage = "mid"
            elif (
                stage_predicted_class == "D"
                and stage_prediction_probability >= self.PREDICTION_PROB_THRESHOLD
            ):
                if self.current_stage in ["init", "mid"]:
                    self.counter += 1
                self.current_stage = "down"

            # Analyze lunge pose - Knee over toe
            k_o_t_error = None
            if self.current_stage == "down":
                err_predicted_class = self.err_model.predict(X)[0]
                err_prediction_probabilities = self.err_model.predict_proba(X)[0]
                err_prediction_probability = round(
                    err_prediction_probabilities[err_prediction_probabilities.argmax()],
                    2,
                )

                if (
                    err_predicted_class == "L"
                    and err_prediction_probability >= self.PREDICTION_PROB_THRESHOLD
                ):
                    k_o_t_error = True
                    self.has_error = True
                elif (
                    err_predicted_class == "C"
                    and err_prediction_probability >= self.PREDICTION_PROB_THRESHOLD
                ):
                    k_o_t_error = False
                    self.has_error = False
            else:
                self.has_error = False

            # Analyze knee angle
            analyzed_results = self.analyze_knee_angle(
                mp_results=mp_results,
                stage=self.current_stage,
                angle_thresholds=self.KNEE_ANGLE_THRESHOLD,
                knee_over_toe=k_o_t_error,
            )

            # Determine if current rep is correct
            if self.current_stage == "down":
                self.has_error = analyzed_results["error"] if not self.has_error else self.has_error

            # Visualization
            # Draw landmarks and connections
            landmark_color = (0, 0, 255) if self.has_error else (0, 255, 0)
            connection_color = (0, 0, 255) if self.has_error else (0, 255, 0)
            
            mp_drawing.draw_landmarks(
                image,
                mp_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=landmark_color, thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=connection_color, thickness=2, circle_radius=1),
            )

            # Create a black translucent overlay for metrics
            overlay = image.copy()
            cv2.rectangle(overlay, (0, 0), (image.shape[1], 40), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

            # Display simplified metrics in white text at top-center
            status_text = "CORRECT" if not self.has_error else "INCORRECT"
            status_color = (255, 255, 255) if not self.has_error else (0, 0, 255)
            
            cv2.putText(
                image,
                f"REPS: {self.counter} | STATUS: {status_text}",
                (image.shape[1]//2 - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                status_color,
                2,
                cv2.LINE_AA,
            )

            # Display knee angles near the knees if available
            if self.right_knee_pos and analyzed_results["right"]["angle"] is not None:
                cv2.putText(
                    image,
                    str(int(analyzed_results["right"]["angle"])),
                    tuple(np.multiply(self.right_knee_pos, video_dimensions).astype(int)),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5,
                    (255, 0, 0) if analyzed_results["right"]["error"] else (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            
            if self.left_knee_pos and analyzed_results["left"]["angle"] is not None:
                cv2.putText(
                    image,
                    str(int(analyzed_results["left"]["angle"])),
                    tuple(np.multiply(self.left_knee_pos, video_dimensions).astype(int)),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5,
                    (255, 0, 0) if analyzed_results["left"]["error"] else (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        except Exception as e:
            print(f"Error while detecting lunge errors: {e}")

def process_video(input_path, output_path):
    """Process video file and save output with progress display"""
    # Initialize video capture
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error opening video file: {input_path}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Initialize MediaPipe Pose with 0.5 detection confidence
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Initialize Lunge Detection
    lunge_detector = LungeDetection()
    
    # Start processing
    start_time = time.time()
    processed_frames = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make detection with 0.5 confidence threshold
        results = pose.process(image)
        
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Lunge detection logic
        if results.pose_landmarks:
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
            lunge_detector.detect(results, image, timestamp)
        
        # Write frame to output video
        out.write(image)
        
        # Update progress
        processed_frames += 1
        progress = (processed_frames / total_frames) * 100
        elapsed_time = time.time() - start_time
        fps = processed_frames / elapsed_time if elapsed_time > 0 else 0
        
        # Update progress in terminal
        sys.stdout.write(f"\rProcessing: {progress:.1f}% | Frames: {processed_frames}/{total_frames} | FPS: {fps:.1f} | Reps: {lunge_detector.counter}")
        sys.stdout.flush()
    
    # Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Print final results
    print(f"\n\nProcessing complete!")
    print(f"Output saved to: {output_path}")
    print(f"Total lunges counted: {lunge_detector.counter}")
    print("Errors detected:")
    for error in lunge_detector.results:
        print(f"- {error['stage']} at {error['timestamp']}ms (rep {error['counter']})")

if __name__ == "__main__":    
    process_video("input.mp4", "output.mp4")