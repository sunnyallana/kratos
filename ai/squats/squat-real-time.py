import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import pickle
from typing import Dict, Any, Optional, Tuple, List

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class SquatAnalyzerRealtime:
    def __init__(self, model_path: str):
        """
        Initialize the real-time squat analyzer
        
        Args:
            model_path: Path to the trained pkl model
        """
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.load_model()
        
        # Counter and stage tracking
        self.counter = 0
        self.current_stage = ""
        
        # Thresholds
        self.PREDICTION_PROB_THRESHOLD = 0.7
        self.VISIBILITY_THRESHOLD = 0.6
        self.FOOT_SHOULDER_RATIO_THRESHOLDS = [1.2, 2.8]
        self.KNEE_FOOT_RATIO_THRESHOLDS = {
            "up": [0.5, 1.0],
            "middle": [0.7, 1.0],
            "down": [0.7, 1.1],
        }
        
        # Warning messages
        self.warnings = []
        
    def load_model(self):
        """Load the trained model and scaler"""
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data.get('feature_names', None)
            print(f"Model loaded successfully from {self.model_path}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def extract_important_keypoints(self, results) -> list:
        """
        Extract important keypoints for squat analysis
        
        Args:
            results: MediaPipe pose results
            
        Returns:
            List of keypoint coordinates
        """
        if not results.pose_landmarks:
            return None
            
        landmarks = results.pose_landmarks.landmark
        
        # Important landmarks for squat detection
        important_landmarks = [
            mp_pose.PoseLandmark.NOSE,
            mp_pose.PoseLandmark.LEFT_SHOULDER,
            mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.LEFT_HIP,
            mp_pose.PoseLandmark.RIGHT_HIP,
            mp_pose.PoseLandmark.LEFT_KNEE,
            mp_pose.PoseLandmark.RIGHT_KNEE,
            mp_pose.PoseLandmark.LEFT_ANKLE,
            mp_pose.PoseLandmark.RIGHT_ANKLE
        ]
        
        row = []
        for landmark in important_landmarks:
            lm = landmarks[landmark.value]
            row.extend([lm.x, lm.y, lm.z, lm.visibility])
        
        return row
    
    @staticmethod
    def calculate_distance(pointX: List[float], pointY: List[float]) -> float:
        """
        Calculate Euclidean distance between 2 points
        
        Args:
            pointX: First point [x, y]
            pointY: Second point [x, y]
            
        Returns:
            Distance between the points
        """
        x1, y1 = pointX
        x2, y2 = pointY
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    def analyze_foot_knee_placement(self, results, stage: str) -> Dict[str, int]:
        """
        Analyze foot and knee placement (following actual code logic)
        
        Args:
            results: MediaPipe pose results
            stage: Current squat stage ('up', 'middle', 'down')
            
        Returns:
            Dictionary with placement evaluations:
                -1: Unknown result due to poor visibility
                0: Correct placement
                1: Placement too tight
                2: Placement too wide
        """
        analyzed_results = {
            "foot_placement": -1,
            "knee_placement": -1,
        }

        if not results.pose_landmarks:
            return analyzed_results

        landmarks = results.pose_landmarks.landmark

        # Visibility check of important landmarks (as in actual code)
        left_foot_index_vis = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].visibility
        right_foot_index_vis = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].visibility
        left_knee_vis = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility
        right_knee_vis = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility

        # If visibility of any keypoints is low cancel the analysis
        if (left_foot_index_vis < self.VISIBILITY_THRESHOLD or 
            right_foot_index_vis < self.VISIBILITY_THRESHOLD or 
            left_knee_vis < self.VISIBILITY_THRESHOLD or 
            right_knee_vis < self.VISIBILITY_THRESHOLD):
            return analyzed_results
        
        # Calculate shoulder width (as in actual code)
        left_shoulder = [
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        ]
        right_shoulder = [
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, 
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
        ]
        shoulder_width = self.calculate_distance(left_shoulder, right_shoulder)

        # Calculate 2-foot width (as in actual code)
        left_foot_index = [
            landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x, 
            landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y
        ]
        right_foot_index = [
            landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x, 
            landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y
        ]
        foot_width = self.calculate_distance(left_foot_index, right_foot_index)

        # Calculate foot and shoulder ratio (as in actual code)
        foot_shoulder_ratio = round(foot_width / shoulder_width, 1)

        # Analyze FOOT PLACEMENT (as in actual code)
        min_ratio_foot_shoulder, max_ratio_foot_shoulder = self.FOOT_SHOULDER_RATIO_THRESHOLDS
        if min_ratio_foot_shoulder <= foot_shoulder_ratio <= max_ratio_foot_shoulder:
            analyzed_results["foot_placement"] = 0
        elif foot_shoulder_ratio < min_ratio_foot_shoulder:
            analyzed_results["foot_placement"] = 1
        elif foot_shoulder_ratio > max_ratio_foot_shoulder:
            analyzed_results["foot_placement"] = 2
        
        # Re-check knee visibility (as in actual code)
        left_knee_vis = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility
        right_knee_vis = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility

        if (left_knee_vis < self.VISIBILITY_THRESHOLD or 
            right_knee_vis < self.VISIBILITY_THRESHOLD):
            return analyzed_results

        # Calculate 2 knee width (as in actual code)
        left_knee = [
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, 
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
        ]
        right_knee = [
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, 
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y
        ]
        knee_width = self.calculate_distance(left_knee, right_knee)

        # Calculate knee and foot ratio (as in actual code)
        knee_foot_ratio = round(knee_width / foot_width, 1)

        # Analyze KNEE placement (as in actual code)
        if stage == "up":
            min_ratio, max_ratio = self.KNEE_FOOT_RATIO_THRESHOLDS.get("up", [0.5, 1.0])
        elif stage == "middle":
            min_ratio, max_ratio = self.KNEE_FOOT_RATIO_THRESHOLDS.get("middle", [0.7, 1.0])
        elif stage == "down":
            min_ratio, max_ratio = self.KNEE_FOOT_RATIO_THRESHOLDS.get("down", [0.7, 1.1])
        else:
            min_ratio, max_ratio = 0, 0  # Unknown stage

        if min_ratio <= knee_foot_ratio <= max_ratio:
            analyzed_results["knee_placement"] = 0
        elif knee_foot_ratio < min_ratio:
            analyzed_results["knee_placement"] = 1
        elif knee_foot_ratio > max_ratio:
            analyzed_results["knee_placement"] = 2
        
        return analyzed_results
    
    def generate_warnings(self, foot_placement: int, knee_placement: int, 
                         prediction_probability: float) -> List[str]:
        """
        Generate warning messages based on analysis (expanded from actual code)
        
        Args:
            foot_placement: Foot placement evaluation code (-1, 0, 1, 2)
            knee_placement: Knee placement evaluation code (-1, 0, 1, 2)
            prediction_probability: Model prediction confidence
            
        Returns:
            List of warning messages
        """
        warnings = []
        
        # Low confidence warning
        if prediction_probability < self.PREDICTION_PROB_THRESHOLD:
            warnings.append("LOW CONFIDENCE - Check posture")
        
        # Foot placement warnings (matching actual code evaluation)
        if foot_placement == 1:
            warnings.append("WARNING: FEET TOO CLOSE")
        elif foot_placement == 2:
            warnings.append("WARNING: FEET TOO WIDE")
        
        # Knee placement warnings (matching actual code evaluation)
        if knee_placement == 1:
            warnings.append("WARNING: KNEES CAVING IN")
        elif knee_placement == 2:
            warnings.append("WARNING: KNEES TOO WIDE")
        
        return warnings
    
    def draw_interface(self, image: np.ndarray, predicted_class: str, 
                      prediction_probability: float, analyzed_results: Dict[str, int]) -> np.ndarray:
        """
        Draw the user interface on the image with extended black title box and centered warnings
        
        Args:
            image: Input image
            predicted_class: Predicted squat stage
            prediction_probability: Prediction confidence
            analyzed_results: Dictionary with foot_placement and knee_placement codes
            
        Returns:
            Image with interface drawn
        """
        height, width = image.shape[:2]
        
        # Create extended black title box (now includes metrics)
        title_box_height = 100  # Increased height to include metrics
        cv2.rectangle(image, (0, 0), (width, title_box_height), (0, 0, 0), -1)
        
        # Add "Squat Analyzer" title in white
        (title_width, title_height), _ = cv2.getTextSize("SQUAT ANALYZER", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        title_x = (width - title_width) // 2
        cv2.putText(image, "SQUAT ANALYZER", (title_x, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Display counter and stage in the extended black area
        counter_text = f"COUNT: {self.counter}"
        stage_text = f"STAGE: {predicted_class.upper()}"
        
        # Prepare counter and stage text
        counter_text = f"COUNT: {self.counter}"
        stage_text = f"STAGE: {predicted_class.upper()}"
        
        # Calculate text sizes
        (counter_width, counter_height), _ = cv2.getTextSize(counter_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        (stage_width, stage_height), _ = cv2.getTextSize(stage_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        
        # Calculate vertical positions to center both lines in the remaining box space
        total_text_height = counter_height + stage_height + 10  # 10px gap between lines
        start_y = title_box_height // 2 + 15  # Start below title
        
        # Draw counter text (centered horizontally and vertically)
        counter_x = (width - counter_width) // 2
        cv2.putText(image, counter_text, (counter_x, start_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Draw stage text (centered horizontally and vertically)
        stage_x = (width - stage_width) // 2
        cv2.putText(image, stage_text, (stage_x, start_y + counter_height + 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Generate warnings based on analysis
        warnings = self.generate_warnings(
            analyzed_results["foot_placement"],
            analyzed_results["knee_placement"],
            prediction_probability
        )
        
        # Display warnings in center of screen (big red text)
        if warnings:
            warning_y = height // 2
            for warning in warnings:
                (text_width, text_height), _ = cv2.getTextSize(warning, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
                text_x = (width - text_width) // 2
                cv2.putText(image, warning, (text_x, warning_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
                warning_y += 60  # Move down for next warning
        
        return image
    
    def run(self):
        """Run the real-time squat analyzer"""
        # Initialize camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Squat Analyzer started. Press 'q' to quit, 'r' to reset counter.")
        
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, image = cap.read()
                
                if not ret:
                    print("Failed to capture frame")
                    break
                
                # Flip image horizontally for mirror effect
                image = cv2.flip(image, 1)
                
                # Convert BGR to RGB for MediaPipe
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                
                # Process pose
                results = pose.process(image_rgb)
                
                # Convert back to BGR
                image_rgb.flags.writeable = True
                image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                
                if results.pose_landmarks:
                    # Draw pose landmarks
                    mp_drawing.draw_landmarks(
                        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(244, 117, 66), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1)
                    )
                    
                    try:
                        # Extract keypoints and make prediction
                        row = self.extract_important_keypoints(results)
                        
                        if row is not None:
                            # Create DataFrame for prediction
                            X = pd.DataFrame([row], columns=self.feature_names)
                            
                            # Scale features
                            X_scaled = self.scaler.transform(X)
                            
                            # Make prediction
                            predicted_class_num = self.model.predict(X_scaled)[0]
                            predicted_class = "down" if predicted_class_num == 0 else "up"
                            
                            # Handle predict_proba if available, otherwise use default confidence
                            if hasattr(self.model, 'predict_proba'):
                                prediction_probabilities = self.model.predict_proba(X_scaled)[0]
                                prediction_probability = prediction_probabilities[prediction_probabilities.argmax()]
                            else:
                                prediction_probability = 1.0  # Default confidence if no proba available
                            
                            # Update counter logic
                            if (predicted_class == "down" and 
                                prediction_probability >= self.PREDICTION_PROB_THRESHOLD):
                                self.current_stage = "down"
                            elif (self.current_stage == "down" and predicted_class == "up" and 
                                  prediction_probability >= self.PREDICTION_PROB_THRESHOLD):
                                self.current_stage = "up"
                                self.counter += 1
                            
                            # Analyze pose
                            analyzed_results = self.analyze_foot_knee_placement(results, self.current_stage)
                            
                            # Draw interface
                            image = self.draw_interface(
                                image, predicted_class, prediction_probability, analyzed_results
                            )
                        
                    except Exception as e:
                        print(f"Error during prediction: {e}")
                        # Draw basic interface with error state
                        error_results = {"foot_placement": -1, "knee_placement": -1}
                        image = self.draw_interface(image, "ERROR", 0.0, error_results)
                
                else:
                    # No pose detected
                    error_results = {"foot_placement": -1, "knee_placement": -1}
                    image = self.draw_interface(image, "NO POSE", 0.0, error_results)
                
                # Display the image
                cv2.imshow('Squat Analyzer - Real-time', image)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.counter = 0
                    self.current_stage = ""
                    print("Counter reset")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Initialize and run the analyzer
    analyzer = SquatAnalyzerRealtime(model_path="./model/squat_lr_model.pkl")
    analyzer.run()