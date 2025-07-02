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

type PlankForm = 'correct' | 'high_back' | 'low_back';

interface ClassificationResult {
  form: PlankForm;
  confidence: number;
}

const PlankClassifier = () => {
  // State variables
  const [isRunning, setIsRunning] = useState(false);
  const [status, setStatus] = useState('Loading...');
  const [classification, setClassification] = useState<ClassificationResult>({ 
    form: 'correct', 
    confidence: 0 
  });
  const [feedback, setFeedback] = useState<string[]>([]);
  const [isMediaPipeLoaded, setIsMediaPipeLoaded] = useState(false);
  const [isTensorFlowLoaded, setIsTensorFlowLoaded] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  
  // Refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const poseRef = useRef<any>(null);
  const modelRef = useRef<tf.LayersModel | null>(null);
  const scalerRef = useRef<{ mean: tf.Tensor; std: tf.Tensor } | null>(null);
  const animationIdRef = useRef<number | null>(null);
  const isRunningRef = useRef(false);
  
  // Constants
  const VISIBILITY_THRESHOLD = 0.65;
  
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
        
        setStatus("Loading AI model...");
        
        // Load the plank classifier model
        try {
          const model = await tf.loadLayersModel('/plank-model/plank_7layer_dropout/model.json');
          modelRef.current = model;
          
          // Load the scaler data
          const scalerResponse = await fetch('/plank-model/input_scaler.json');
          const scalerData = await scalerResponse.json();
          
          scalerRef.current = {
            mean: tf.tensor(scalerData.mean),
            std: tf.tensor(scalerData.std)
          };
          
        } catch (error) {
          console.error("Error loading model:", error);
          // Continue without model - we can still do basic pose detection
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
            width: 1280, 
            height: 720,
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
    const ctx = canvas?.getContext('2d');
    
    if (!ctx || !video || !canvas) return;
    
    // Set canvas size to match video
    canvas.width = video.videoWidth || 1280;
    canvas.height = video.videoHeight || 720;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    if (results.poseLandmarks) {
      // Draw pose landmarks using MediaPipe drawing utils
      if (window.drawConnectors && window.drawLandmarks && window.POSE_CONNECTIONS) {
        window.drawConnectors(ctx, results.poseLandmarks, window.POSE_CONNECTIONS, {
          color: '#00FF00',
          lineWidth: 2
        });
        window.drawLandmarks(ctx, results.poseLandmarks, {
          color: '#FF0000',
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
      setClassification({ form: 'correct', confidence: 0 });
      setFeedback(["Position yourself in the camera view"]);
    }
  };

  // Fallback method to draw pose landmarks
  const drawPoseLandmarks = (ctx: CanvasRenderingContext2D, landmarks: any[]) => {
    // Draw connections
    const connections = [
      [POSE_LANDMARKS.LEFT_SHOULDER, POSE_LANDMARKS.RIGHT_SHOULDER],
      [POSE_LANDMARKS.LEFT_SHOULDER, POSE_LANDMARKS.LEFT_ELBOW],
      [POSE_LANDMARKS.LEFT_ELBOW, POSE_LANDMARKS.LEFT_WRIST],
      [POSE_LANDMARKS.RIGHT_SHOULDER, POSE_LANDMARKS.RIGHT_ELBOW],
      [POSE_LANDMARKS.RIGHT_ELBOW, POSE_LANDMARKS.RIGHT_WRIST],
      [POSE_LANDMARKS.LEFT_SHOULDER, POSE_LANDMARKS.LEFT_HIP],
      [POSE_LANDMARKS.RIGHT_SHOULDER, POSE_LANDMARKS.RIGHT_HIP],
      [POSE_LANDMARKS.LEFT_HIP, POSE_LANDMARKS.RIGHT_HIP],
      [POSE_LANDMARKS.LEFT_HIP, POSE_LANDMARKS.LEFT_KNEE],
      [POSE_LANDMARKS.RIGHT_HIP, POSE_LANDMARKS.RIGHT_KNEE],
      [POSE_LANDMARKS.LEFT_KNEE, POSE_LANDMARKS.LEFT_ANKLE],
      [POSE_LANDMARKS.RIGHT_KNEE, POSE_LANDMARKS.RIGHT_ANKLE],
    ];
    
    ctx.strokeStyle = '#10b981';
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

  // Process landmarks for plank classification
  const processLandmarks = async (landmarks: any[]) => {
    setIsProcessing(true);
    
    try {
      if (!modelRef.current || !scalerRef.current) {
        return;
      }
      
      // Preprocess landmarks
      const inputTensor = preprocessLandmarks(landmarks);
      if (!inputTensor) {
        return;
      }
      
      // Make prediction
      const prediction = modelRef.current.predict(inputTensor) as tf.Tensor;
      const predictionData = await prediction.data();
      
      // Interpret results (0 = Correct, 1 = High back, 2 = Low back)
      const maxProbability = Math.max(...predictionData);
      const predictedClass = predictionData.indexOf(maxProbability);
      const confidence = maxProbability;
      
      let form: PlankForm = 'correct';
      if (predictedClass === 1) form = 'high_back';
      if (predictedClass === 2) form = 'low_back';
      
      setClassification({
        form,
        confidence
      });
      
      // Generate feedback
      generateFeedback(form, landmarks);
      
      // Clean up tensors
      tf.dispose([prediction, inputTensor]);
      
    } catch (error) {
      console.error("Processing error:", error);
      setClassification({ form: 'correct', confidence: 0 });
      setFeedback(["Error processing pose"]);
    } finally {
      setIsProcessing(false);
    }
  };

  // Preprocess landmarks for TensorFlow model
  const preprocessLandmarks = (landmarks: any[]): tf.Tensor2D | null => {
    try {
      const importantLandmarks = [
        "NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", 
        "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIP", 
        "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", 
        "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX", 
        "RIGHT_FOOT_INDEX"
      ];
      
      const orderedFeatures: number[] = [];
      
      importantLandmarks.forEach(landmarkName => {
        const landmarkIndex = POSE_LANDMARKS[landmarkName as keyof typeof POSE_LANDMARKS];
        const landmark = landmarks[landmarkIndex];
        
        if (landmark) {
          orderedFeatures.push(landmark.x, landmark.y, landmark.z, landmark.visibility);
        } else {
          // Fill with zeros if landmark not detected
          orderedFeatures.push(0, 0, 0, 0);
        }
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

  // Generate feedback based on plank form
  const generateFeedback = (form: PlankForm, landmarks: any[]) => {
    const feedbackMessages: string[] = [];
    
    if (form === 'correct') {
    } else if (form === 'high_back') {
      feedbackMessages.push("Your hips are too high (butt up in the air)");
      feedbackMessages.push("Try to align your shoulders, hips and ankles in a straight line");
    } else if (form === 'low_back') {
      feedbackMessages.push("Your back is sagging (hips too low)");
      feedbackMessages.push("Engage your core muscles to keep your body straight");
    }
    
    // Additional alignment checks
    const leftShoulder = landmarks[POSE_LANDMARKS.LEFT_SHOULDER];
    const rightShoulder = landmarks[POSE_LANDMARKS.RIGHT_SHOULDER];
    const leftHip = landmarks[POSE_LANDMARKS.LEFT_HIP];
    const rightHip = landmarks[POSE_LANDMARKS.RIGHT_HIP];
    
    if (leftShoulder && rightShoulder && 
        Math.abs(leftShoulder.y - rightShoulder.y) > 0.05) {
      feedbackMessages.push("Your shoulders are uneven. Try to balance them");
    }
    
    if (leftHip && rightHip && 
        Math.abs(leftHip.y - rightHip.y) > 0.05) {
      feedbackMessages.push("Your hips are uneven. Try to balance them");
    }
    
    setFeedback(feedbackMessages);
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
      </div>

      {/* Form Feedback */}
      <div className="absolute top-20 left-4 max-w-xs">
        {feedback.length > 0 && (
          <div className="space-y-2">
            {feedback.map((message, index) => (
              <div key={index} className={`${
                message.startsWith("Great") 
                  ? "bg-green-50 border-l-4 border-green-400 text-green-800" 
                  : "bg-amber-50 border-l-4 border-amber-400 text-amber-800"
              } p-3 rounded flex items-start gap-2`}>
                {message.startsWith("Great") ? (
                  <CheckCircle className="w-4 h-4 text-green-600 mt-0.5 flex-shrink-0" />
                ) : (
                  <AlertTriangle className="w-4 h-4 text-amber-600 mt-0.5 flex-shrink-0" />
                )}
                <div className="font-medium text-sm">{message}</div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default PlankClassifier;