import { useState, useEffect, useRef } from 'react';
import { RefreshCw, Play, Square, Activity, AlertTriangle, CheckCircle } from 'lucide-react';
import * as tf from '@tensorflow/tfjs';

declare global {
  interface Window {
    Pose: any;
    Camera: any;
    drawConnectors: any;
    drawLandmarks: any;
    POSE_CONNECTIONS: any;
  }
}

interface ClassificationResult {
  stage: 'init' | 'mid' | 'down';
  confidence: number;
  error: 'correct' | 'knee_collapse' | 'back_bend' | 'shallow' | null;
}

interface FormFeedback {
  knees: 'good' | 'poor';
  back: 'good' | 'poor';
}

const LungeClassifier = () => {
  // State variables
  const [isRunning, setIsRunning] = useState(false);
  const [status, setStatus] = useState('Loading...');
  const [counter, setCounter] = useState(0);
  const [classification, setClassification] = useState<ClassificationResult>({ 
    stage: 'init',
    confidence: 0,
    error: null
  });
  const [formFeedback, setFormFeedback] = useState<FormFeedback>({
    knees: 'good',
    back: 'good'
  });
  const [warnings, setWarnings] = useState<string[]>([]);
  const [isMediaPipeLoaded, setIsMediaPipeLoaded] = useState(false);
  const [isTensorFlowLoaded, setIsTensorFlowLoaded] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  
  // Refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const poseRef = useRef<any>(null);
  const stageModelRef = useRef<tf.LayersModel | null>(null);
  const errorModelRef = useRef<tf.LayersModel | null>(null);
  const scalerRef = useRef<{ mean: tf.Tensor; std: tf.Tensor } | null>(null);
  const stageRef = useRef<'init' | 'mid' | 'down'>('init');
  const animationIdRef = useRef<number | null>(null);
  const isRunningRef = useRef(false);
  
  // Constants
  const KNEE_ANGLE_THRESHOLD = [60, 125];
  const BACK_ANGLE_THRESHOLD = 20;
  const VISIBILITY_THRESHOLD = 0.65;
  const PREDICTION_PROB_THRESHOLD = 0.8;
  
  // MediaPipe pose landmarks mapping
  const POSE_LANDMARKS = {
    NOSE: 0,
    LEFT_SHOULDER: 11,
    RIGHT_SHOULDER: 12,
    LEFT_ELBOW: 13,
    RIGHT_ELBOW: 14,
    LEFT_WRIST: 15,
    RIGHT_WRIST: 16,
    LEFT_HIP: 23,
    RIGHT_HIP: 24,
    LEFT_KNEE: 25,
    RIGHT_KNEE: 26,
    LEFT_ANKLE: 27,
    RIGHT_ANKLE: 28,
    LEFT_HEEL: 29,
    RIGHT_HEEL: 30,
    LEFT_FOOT_INDEX: 31,
    RIGHT_FOOT_INDEX: 32
  };

  // Initialize everything
  useEffect(() => {
    const initializeApp = async () => {
      try {
        setStatus("Loading TensorFlow.js...");
        
        // Wait for TensorFlow.js to be ready
        await tf.ready();
        setIsTensorFlowLoaded(true);
        
        setStatus("Loading AI models...");
        
        // Load the lunge classifier models
        try {
          const stageModel = await tf.loadLayersModel('/lunge-model/lunge_stage_model/model.json');
          stageModelRef.current = stageModel;
          
          const errorModel = await tf.loadLayersModel('/lunge-model/lunge_error_model/model.json');
          errorModelRef.current = errorModel;
          
          // Load the scaler data
          const scalerResponse = await fetch('/lunge-model/lunge_stage_model_metadata.json');
          const scalerData = await scalerResponse.json();
          
          scalerRef.current = {
            mean: tf.tensor(scalerData.scaler.mean),
            std: tf.tensor(scalerData.scaler.std)
          };
          
        } catch (error) {
          console.error("Error loading models:", error);
          setStatus("Error loading models. See console for details.");
          return;
        }
        
        setStatus("Checking MediaPipe availability...");
        
        // Wait for MediaPipe to be loaded
        let attempts = 0;
        while (!window.Pose && attempts < 50) {
          await new Promise(resolve => setTimeout(resolve, 100));
          attempts++;
        }
        
        if (!window.Pose) {
          throw new Error("MediaPipe Pose not loaded. Please refresh the page.");
        }
        
        setIsMediaPipeLoaded(true);
        
        setStatus("Initializing camera...");
        
        // Initialize camera
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { 
            width: { ideal: 1280 },
            height: { ideal: 720 },
            facingMode: "user" 
          }
        });
        
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await new Promise((resolve) => {
            if (videoRef.current) {
              videoRef.current.onloadedmetadata = () => {
                if (videoRef.current) {
                  videoRef.current.play();
                }
                resolve(null);
              };
            }
          });
        }
        
        setStatus("Loading pose detection model...");
        
        // Initialize MediaPipe Pose
        const pose = new window.Pose({
          locateFile: (file: string) => {
            return `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`;
          }
        });
        
        pose.setOptions({
          modelComplexity: 1,
          smoothLandmarks: true,
          enableSegmentation: false,
          smoothSegmentation: false,
          minDetectionConfidence: 0.5,
          minTrackingConfidence: 0.5
        });
        
        pose.onResults(onPoseResults);
        poseRef.current = pose;
        
        setStatus("Ready! Click 'Start Detection' to begin.");
        
      } catch (error) {
        console.error("Initialization error:", error);
        const errorMessage = error instanceof Error ? error.message : 'Unknown error';
        setStatus(`Error: ${errorMessage}`);
      }
    };
    
    initializeApp();
    
    // Cleanup
    return () => {
      if (animationIdRef.current) {
        cancelAnimationFrame(animationIdRef.current);
      }
      if (videoRef.current && videoRef.current.srcObject) {
        const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
        tracks.forEach(track => track.stop());
      }
      // Clean up TensorFlow tensors
      if (scalerRef.current) {
        scalerRef.current.mean.dispose();
        scalerRef.current.std.dispose();
      }
    };
  }, []);

  // Classification loop
  const classifyFrame = async () => {
    if (!isRunningRef.current || !poseRef.current || !videoRef.current) {
      return;
    }
    
    if (videoRef.current.readyState < 2) {
      if (isRunningRef.current) {
        animationIdRef.current = requestAnimationFrame(classifyFrame);
      }
      return;
    }
    
    try {
      await poseRef.current.send({ image: videoRef.current });
    } catch (error) {
      console.error("Classification error:", error);
    }
    
    if (isRunningRef.current) {
      animationIdRef.current = requestAnimationFrame(classifyFrame);
    }
  };

  // Process pose results from MediaPipe
  const onPoseResults = (results: any) => {
    if (!isRunningRef.current) return;
    
    const canvas = canvasRef.current;
    const video = videoRef.current;
    
    if (!canvas || !video) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    if (results.poseLandmarks) {
      // Draw pose landmarks using MediaPipe drawing utils
      if (window.drawConnectors && window.drawLandmarks && window.POSE_CONNECTIONS) {
        window.drawConnectors(ctx, results.poseLandmarks, window.POSE_CONNECTIONS, {
          color: classification.error ? '#FF0000' : '#00FF00',
          lineWidth: 2
        });
        window.drawLandmarks(ctx, results.poseLandmarks, {
          color: classification.error ? '#FF0000' : '#00FF00',
          lineWidth: 1,
          radius: 3
        });
      } else {
        // Fallback drawing method
        drawPoseLandmarks(ctx, results.poseLandmarks);
      }
      
      // Process landmarks for classification
      processLandmarks(results.poseLandmarks);
    } else {
      // No human detected
      ctx.font = '24px Arial';
      ctx.fillStyle = '#ef4444';
      ctx.textAlign = 'center';
      ctx.fillText("NO POSE DETECTED", canvas.width/2, canvas.height/2);
      ctx.fillText("Position yourself in front of the camera", canvas.width/2, canvas.height/2 + 30);
      setWarnings(["Position yourself in the camera view"]);
    }
  };

  // Fallback method to draw pose landmarks
  const drawPoseLandmarks = (ctx: CanvasRenderingContext2D, landmarks: any[]) => {
    // Draw connections
    const connections = [
      [POSE_LANDMARKS.LEFT_SHOULDER, POSE_LANDMARKS.RIGHT_SHOULDER],
      [POSE_LANDMARKS.LEFT_SHOULDER, POSE_LANDMARKS.LEFT_HIP],
      [POSE_LANDMARKS.RIGHT_SHOULDER, POSE_LANDMARKS.RIGHT_HIP],
      [POSE_LANDMARKS.LEFT_HIP, POSE_LANDMARKS.RIGHT_HIP],
      [POSE_LANDMARKS.LEFT_HIP, POSE_LANDMARKS.LEFT_KNEE],
      [POSE_LANDMARKS.RIGHT_HIP, POSE_LANDMARKS.RIGHT_KNEE],
      [POSE_LANDMARKS.LEFT_KNEE, POSE_LANDMARKS.LEFT_ANKLE],
      [POSE_LANDMARKS.RIGHT_KNEE, POSE_LANDMARKS.RIGHT_ANKLE],
    ];
    
    ctx.strokeStyle = classification.error ? '#FF0000' : '#10b981';
    ctx.lineWidth = 3;
    
    connections.forEach(([start, end]) => {
      const startPoint = landmarks[start];
      const endPoint = landmarks[end];
      
      if (startPoint && endPoint && startPoint.visibility > VISIBILITY_THRESHOLD && endPoint.visibility > VISIBILITY_THRESHOLD) {
        ctx.beginPath();
        ctx.moveTo(startPoint.x * ctx.canvas.width, startPoint.y * ctx.canvas.height);
        ctx.lineTo(endPoint.x * ctx.canvas.width, endPoint.y * ctx.canvas.height);
        ctx.stroke();
      }
    });
  };

  // Process landmarks for lunge classification
  const processLandmarks = async (landmarks: any[]) => {
    setIsProcessing(true);
    
    try {
      if (!stageModelRef.current || !errorModelRef.current || !scalerRef.current) {
        return;
      }
      
      // 1. Extract important landmarks
      const importantLandmarks = [
        "NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP",
        "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE",
        "LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
      ];
      
      const landmarkData: Record<string, number> = {};
      importantLandmarks.forEach(lm => {
        const landmark = landmarks[POSE_LANDMARKS[lm as keyof typeof POSE_LANDMARKS]];
        if (landmark) {
          const key = lm.toLowerCase();
          landmarkData[`${key}_x`] = landmark.x;
          landmarkData[`${key}_y`] = landmark.y;
          landmarkData[`${key}_z`] = landmark.z;
          landmarkData[`${key}_v`] = landmark.visibility;
        }
      });
      
      // 2. Preprocess the landmarks for the model
      const inputTensor = preprocessLandmarks(landmarkData);
      if (!inputTensor) return;
      
      // 3. Make stage prediction
      const stagePrediction = stageModelRef.current.predict(inputTensor) as tf.Tensor;
      const stageProbabilities = await stagePrediction.array() as number[][];
      const stageConfidence = Math.max(...stageProbabilities[0]);
      const predictedStage = stageProbabilities[0].indexOf(stageConfidence);
      
      let newStage: 'init' | 'mid' | 'down' = 'init';
      let newError: 'correct' | 'knee_collapse' | 'back_bend' | 'shallow' | null = null;
      
      // Only update stage if confidence is high enough
      if (stageConfidence >= PREDICTION_PROB_THRESHOLD) {
        newStage = ['init', 'mid', 'down'][predictedStage] as 'init' | 'mid' | 'down';
        
        // Update counter when transitioning from down to init
        if (newStage === 'init' && stageRef.current === 'down') {
          setCounter(prev => prev + 1);
        }
        
        stageRef.current = newStage;
      }
      
      // 4. Make error prediction (only in down stage)
      if (newStage === 'down') {
        const errPrediction = errorModelRef.current.predict(inputTensor) as tf.Tensor;
        const errProbabilities = await errPrediction.array() as number[][];
        const errConfidence = Math.max(...errProbabilities[0]);
        
        if (errConfidence >= PREDICTION_PROB_THRESHOLD) {
          newError = ['correct', 'knee_collapse', 'back_bend', 'shallow'][
            errProbabilities[0].indexOf(errConfidence)
          ] as 'correct' | 'knee_collapse' | 'back_bend' | 'shallow';
        }
        
        tf.dispose([errPrediction]);
      }
      
      // 5. Update classification state
      setClassification({
        stage: newStage,
        confidence: stageConfidence,
        error: newError
      });
      
      // 6. Analyze form for feedback
      analyzeForm(landmarks);
      
      // 7. Clean up
      tf.dispose([stagePrediction, inputTensor]);
      
    } catch (error) {
      console.error("Processing error:", error);
      setWarnings(["Error processing pose"]);
    } finally {
      setIsProcessing(false);
    }
  };

  // Preprocess landmarks for TensorFlow model
  const preprocessLandmarks = (landmarks: Record<string, number>): tf.Tensor2D | null => {
    try {
      const LUNGE_LANDMARKS = [
        "NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP",
        "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE",
        "LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
      ];
      
      const orderedFeatures: number[] = [];
      
      LUNGE_LANDMARKS.forEach(landmark => {
        const key = landmark.toLowerCase();
        orderedFeatures.push(
          landmarks[`${key}_x`] || 0,
          landmarks[`${key}_y`] || 0,
          landmarks[`${key}_z`] || 0,
          landmarks[`${key}_v`] || 0
        );
      });
      
      // Convert to tensor and normalize using the scaler
      const featuresTensor = tf.tensor2d([orderedFeatures]);
      const normalizedFeatures = featuresTensor.sub(scalerRef.current!.mean).div(scalerRef.current!.std);
      featuresTensor.dispose();
      
      return normalizedFeatures as tf.Tensor2D;
    } catch (error) {
      console.error("Preprocessing error:", error);
      return null;
    }
  };

  // Analyze lunge form for feedback
  const analyzeForm = (landmarks: any[]) => {
    const newWarnings: string[] = [];
    const newFormFeedback: FormFeedback = {
      knees: 'good',
      back: 'good'
    };
    
    try {
      // Get relevant landmarks
      const leftHip = landmarks[POSE_LANDMARKS.LEFT_HIP];
      const rightHip = landmarks[POSE_LANDMARKS.RIGHT_HIP];
      const leftKnee = landmarks[POSE_LANDMARKS.LEFT_KNEE];
      const rightKnee = landmarks[POSE_LANDMARKS.RIGHT_KNEE];
      const leftAnkle = landmarks[POSE_LANDMARKS.LEFT_ANKLE];
      const rightAnkle = landmarks[POSE_LANDMARKS.RIGHT_ANKLE];
      const leftShoulder = landmarks[POSE_LANDMARKS.LEFT_SHOULDER];
      const rightShoulder = landmarks[POSE_LANDMARKS.RIGHT_SHOULDER];
      
      // Check visibility
      if (!leftHip || !rightHip || !leftKnee || !rightKnee || 
          leftHip.visibility < VISIBILITY_THRESHOLD || 
          rightHip.visibility < VISIBILITY_THRESHOLD || 
          leftKnee.visibility < VISIBILITY_THRESHOLD || 
          rightKnee.visibility < VISIBILITY_THRESHOLD) {
        return;
      }
      
      // Calculate knee angles
      const leftKneeAngle = calculateAngle(
        [leftHip.x, leftHip.y],
        [leftKnee.x, leftKnee.y],
        [leftAnkle.x, leftAnkle.y]
      );
      
      const rightKneeAngle = calculateAngle(
        [rightHip.x, rightHip.y],
        [rightKnee.x, rightKnee.y],
        [rightAnkle.x, rightAnkle.y]
      );
      
      // Calculate back angle (shoulder to hip angle from vertical)
      const backAngle = calculateAngle(
        [leftShoulder.x, leftShoulder.y],
        [leftHip.x, leftHip.y],
        [leftHip.x, 0] // Vertical reference
      );
      
      // Check for knee collapse (poor form)
      if (leftKneeAngle < KNEE_ANGLE_THRESHOLD[0] || rightKneeAngle < KNEE_ANGLE_THRESHOLD[0]) {
        newWarnings.push("KNEE ANGLE TOO SHALLOW");
        newFormFeedback.knees = 'poor';
      } else if (leftKneeAngle > KNEE_ANGLE_THRESHOLD[1] || rightKneeAngle > KNEE_ANGLE_THRESHOLD[1]) {
        newWarnings.push("KNEE ANGLE TOO DEEP");
        newFormFeedback.knees = 'poor';
      }
      
      // Check for back angle (poor form)
      if (backAngle > BACK_ANGLE_THRESHOLD) {
        newWarnings.push("BACK NOT STRAIGHT");
        newFormFeedback.back = 'poor';
      }
      
      // Add error warning if present
      if (classification.error && classification.error !== 'correct') {
        newWarnings.push(classification.error.toUpperCase().replace('_', ' '));
      }
      
    } catch (error) {
      console.error("Error analyzing form:", error);
      newWarnings.push("Error analyzing form");
    }
    
    setWarnings(newWarnings);
    setFormFeedback(newFormFeedback);
  };

  // Calculate angle between 3 points
  const calculateAngle = (point1: number[], point2: number[], point3: number[]): number => {
    const vector1 = [point1[0] - point2[0], point1[1] - point2[1]];
    const vector2 = [point3[0] - point2[0], point3[1] - point2[1]];
    
    const dotProduct = vector1[0] * vector2[0] + vector1[1] * vector2[1];
    const magnitude1 = Math.sqrt(vector1[0] ** 2 + vector1[1] ** 2);
    const magnitude2 = Math.sqrt(vector2[0] ** 2 + vector2[1] ** 2);
    
    if (magnitude1 === 0 || magnitude2 === 0) return 0;
    
    const angleRad = Math.acos(Math.max(-1, Math.min(1, dotProduct / (magnitude1 * magnitude2))));
    let angleDeg = angleRad * (180 / Math.PI);
    
    return angleDeg > 180 ? 360 - angleDeg : angleDeg;
  };

  // Toggle classification
  const toggleClassification = () => {
    if (!isMediaPipeLoaded) {
      setStatus("MediaPipe not loaded yet. Please wait or refresh the page.");
      return;
    }
    
    const newRunningState = !isRunning;
    setIsRunning(newRunningState);
    isRunningRef.current = newRunningState;
    
    if (newRunningState) {
      setStatus("Running AI-powered pose detection...");
      classifyFrame();
    } else {
      setStatus("Stopped");
      if (animationIdRef.current) {
        cancelAnimationFrame(animationIdRef.current);
      }
    }
  };

  // Reset counter
  const resetCounter = () => {
    setCounter(0);
    stageRef.current = 'init';
  };

  return (
    <div className="relative bg-black rounded-xl overflow-hidden shadow-2xl">
      <video
        ref={videoRef}
        className="w-full block"
        autoPlay
        muted
        playsInline
        style={{ maxHeight: '600px', transform: 'scaleX(1)' }}
      />
      <canvas
        ref={canvasRef}
        className="absolute top-0 left-0 w-full h-full"
        style={{ transform: 'scaleX(1)' }}
      />
      
      {/* Status overlay */}
      <div className="absolute top-4 left-4 bg-black bg-opacity-50 text-white px-3 py-2 rounded-lg">
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4" />
          <span className="text-sm font-medium">{status}</span>
        </div>
      </div>
      
      {/* AI Classification overlay */}
      {isRunning && (
        <div className="absolute top-4 right-4 space-y-2">
          <div className="space-y-1">
            <div className={`px-3 py-1 rounded-lg text-sm font-medium ${
              classification.stage === 'init' 
                ? 'bg-blue-500 bg-opacity-80 text-white' 
                : classification.stage === 'mid'
                ? 'bg-yellow-500 bg-opacity-80 text-white'
                : 'bg-purple-500 bg-opacity-80 text-white'
            }`}>
              {classification.stage === 'init' ? 'INITIAL POSITION' : 
               classification.stage === 'mid' ? 'MID POSITION' : 'DOWN POSITION'}
              <span className="text-xs ml-2">({Math.round(classification.confidence * 100)}%)</span>
            </div>
          </div>
          
          <div className="bg-indigo-500 bg-opacity-80 text-white px-3 py-1 rounded-lg text-sm">
            Lunges: {counter}
          </div>
          
          {isProcessing && (
            <div className="bg-yellow-500 bg-opacity-80 text-white px-3 py-1 rounded-lg text-xs">
              Processing...
            </div>
          )}
        </div>
      )}

      {/* Controls */}
      <div className="absolute bottom-4 left-0 right-0 flex justify-center gap-4">
        <button
          onClick={toggleClassification}
          disabled={!isMediaPipeLoaded || !isTensorFlowLoaded}
          className={`px-6 py-3 rounded-lg font-semibold transition-all duration-200 flex items-center justify-center gap-2 ${
            !isMediaPipeLoaded || !isTensorFlowLoaded
              ? 'bg-gray-300 cursor-not-allowed text-gray-500' 
              : isRunning
              ? 'bg-red-500 hover:bg-red-600 text-white shadow-lg hover:shadow-xl'
              : 'bg-green-500 hover:bg-green-600 text-white shadow-lg hover:shadow-xl'
          }`}
        >
          {isRunning ? (
            <>
              <Square className="w-5 h-5" />
              Stop
            </>
          ) : (
            <>
              <Play className="w-5 h-5" />
              Start
            </>
          )}
        </button>
        
        <button
          onClick={resetCounter}
          className="px-6 py-3 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg transition-colors font-medium flex items-center justify-center gap-2"
        >
          <RefreshCw className="w-4 h-4" />
          Reset
        </button>
      </div>

      {/* Form Feedback */}
      <div className="absolute bottom-24 left-0 right-0 flex justify-center gap-4">
        <div className={`px-4 py-2 rounded-lg ${
          formFeedback.knees === 'good' 
            ? 'bg-green-100 text-green-800' 
            : 'bg-red-100 text-red-800'
        }`}>
          {formFeedback.knees === 'good' ? 'KNEE POSITION: GOOD' : 'KNEE POSITION: POOR'}
        </div>
        <div className={`px-4 py-2 rounded-lg ${
          formFeedback.back === 'good' 
            ? 'bg-green-100 text-green-800' 
            : 'bg-red-100 text-red-800'
        }`}>
          {formFeedback.back === 'good' ? 'BACK POSITION: GOOD' : 'BACK POSITION: POOR'}
        </div>
      </div>

      {/* Warnings */}
      <div className="absolute top-32 left-4 right-4">
        {warnings.length > 0 && (
          <div className="space-y-2">
            {warnings.map((warning, index) => (
              <div key={index} className="bg-amber-50 border-l-4 border-amber-400 p-3 rounded flex items-start gap-2">
                <AlertTriangle className="w-4 h-4 text-amber-600 mt-0.5 flex-shrink-0" />
                <div className="text-amber-800 font-medium text-sm">{warning}</div>
              </div>    
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default LungeClassifier;