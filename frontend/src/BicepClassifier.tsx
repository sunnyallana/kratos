import { useState, useEffect, useRef } from 'react';
import { RefreshCw, Play, Square, Activity, Eye, EyeOff, Brain, CheckCircle, AlertTriangle } from 'lucide-react';
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
  class: 'correct' | 'low_form';
  confidence: number;
}

interface ArmAnalysis {
  counter: number;
  warnings: string[];
  angle: number;
  stage: 'up' | 'down';
}

const BicepClassifier = () => {
  // State variables
  const [isRunning, setIsRunning] = useState(false);
  const [status, setStatus] = useState('Loading...');
  const [leftCounter, setLeftCounter] = useState(0);
  const [rightCounter, setRightCounter] = useState(0);
  const [currentWarnings, setCurrentWarnings] = useState<string[]>([]);
  const [debugInfo, setDebugInfo] = useState('');
  const [isMediaPipeLoaded, setIsMediaPipeLoaded] = useState(false);
  const [isTensorFlowLoaded, setIsTensorFlowLoaded] = useState(false);
  const [showDebug, setShowDebug] = useState(true);
  const [leftAngle, setLeftAngle] = useState(0);
  const [rightAngle, setRightAngle] = useState(0);
  const [classification, setClassification] = useState<ClassificationResult>({ class: 'correct', confidence: 0 });
  const [isProcessing, setIsProcessing] = useState(false);
  
  // Refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const poseRef = useRef<any>(null);
  const modelRef = useRef<tf.LayersModel | null>(null);
  const scalerRef = useRef<{ mean: tf.Tensor; std: tf.Tensor } | null>(null);
  const leftStageRef = useRef<'up' | 'down'>('down');
  const rightStageRef = useRef<'up' | 'down'>('down');
  const leftCounterRef = useRef(0);
  const rightCounterRef = useRef(0);
  const animationIdRef = useRef<number | null>(null);
  const isRunningRef = useRef(false);
  
  // Constants
  const STAGE_UP_THRESHOLD = 90;
  const STAGE_DOWN_THRESHOLD = 120;
  const LOOSE_UPPER_ARM_THRESHOLD = 40;
  const PEAK_CONTRACTION_THRESHOLD = 60;
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
  };

  // Initialize everything
  useEffect(() => {
    const initializeApp = async () => {
      try {
        setStatus("Loading TensorFlow.js...");
        setDebugInfo("Initializing TensorFlow.js backend...");
        
        // Wait for TensorFlow.js to be ready
        await tf.ready();
        setIsTensorFlowLoaded(true);
        setDebugInfo("TensorFlow.js ready");
        
        setStatus("Loading AI model...");
        
        // Load the bicep classifier model
        try {
          const model = await tf.loadLayersModel('/bicep-curl-model/bicep_classifier/model.json');
          modelRef.current = model;
          setDebugInfo("Bicep classifier model loaded");
          
          // Load the scaler data
          const scalerResponse = await fetch('/bicep-curl-model/input_scaler.json');
          const scalerData = await scalerResponse.json();
          
          scalerRef.current = {
            mean: tf.tensor(scalerData.mean),
            std: tf.tensor(scalerData.std)
          };
          setDebugInfo("Model and scaler loaded successfully");
          
        } catch (error) {
          console.error("Error loading model:", error);
          setDebugInfo("Model loading failed - using pose-only mode");
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
        setDebugInfo("MediaPipe loaded successfully");
        
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
          setDebugInfo("Camera initialized successfully");
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
        setDebugInfo("All systems ready");
        
      } catch (error) {
        console.error("Initialization error:", error);
        const errorMessage = error instanceof Error ? error.message : 'Unknown error';
        setStatus(`Error: ${errorMessage}`);
        setDebugInfo(`Error: ${errorMessage}`);
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
    if (!isRunningRef.current) {
      setDebugInfo("Classification stopped - not running");
      return;
    }
    
    if (!poseRef.current) {
      setDebugInfo("Classification stopped - MediaPipe pose not initialized");
      return;
    }
    
    if (!videoRef.current) {
      setDebugInfo("Classification stopped - Video reference not available");
      return;
    }
    
    if (videoRef.current.readyState < 2) {
      setDebugInfo(`Video not ready (readyState: ${videoRef.current.readyState}), retrying...`);
      if (isRunningRef.current) {
        animationIdRef.current = requestAnimationFrame(classifyFrame);
      }
      return;
    }
    
    try {
      setDebugInfo(`Sending frame to MediaPipe... (${new Date().toLocaleTimeString()})`);
      await poseRef.current.send({ image: videoRef.current });
    } catch (error) {
      console.error("Classification error:", error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      setDebugInfo(`Classification error: ${errorMessage}`);
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
      setDebugInfo(`Pose detected with ${results.poseLandmarks.length} landmarks`);
      
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
      
      // Process landmarks for classification and counting
      processLandmarks(results.poseLandmarks);
    } else {
      // No human detected
      ctx.font = '24px Arial';
      ctx.fillStyle = '#ef4444';
      ctx.textAlign = 'center';
      ctx.fillText("NO POSE DETECTED", canvas.width/2, canvas.height/2);
      ctx.fillText("Position yourself in front of the camera", canvas.width/2, canvas.height/2 + 30);
      setDebugInfo("No pose landmarks detected");
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
    
    // Draw key landmarks
    const keyLandmarks = [
      POSE_LANDMARKS.LEFT_SHOULDER,
      POSE_LANDMARKS.RIGHT_SHOULDER,
      POSE_LANDMARKS.LEFT_ELBOW,
      POSE_LANDMARKS.RIGHT_ELBOW,
      POSE_LANDMARKS.LEFT_WRIST,
      POSE_LANDMARKS.RIGHT_WRIST,
    ];
    
    keyLandmarks.forEach((index) => {
      const landmark = landmarks[index];
      if (landmark && landmark.visibility > VISIBILITY_THRESHOLD) {
        ctx.beginPath();
        ctx.arc(
          landmark.x * ctx.canvas.width,
          landmark.y * ctx.canvas.height,
          8, 0, 2 * Math.PI
        );
        ctx.fillStyle = '#ef4444';
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    });
  };

  // Preprocess landmarks for TensorFlow model
  const preprocessLandmarks = (landmarks: any[]): tf.Tensor2D | null => {
    try {
      const importantLandmarks = [
        "NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "RIGHT_ELBOW", 
        "LEFT_ELBOW", "RIGHT_WRIST", "LEFT_WRIST", "LEFT_HIP", "RIGHT_HIP"
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
      
      if (orderedFeatures.length !== 36) {
        console.warn(`Expected 36 features, got ${orderedFeatures.length}`);
        return null;
      }
      
      // Convert to tensor and normalize using the scaler
      const featuresTensor = tf.tensor2d([orderedFeatures]);
      
      if (scalerRef.current) {
        const normalizedFeatures = featuresTensor.sub(scalerRef.current.mean).div(scalerRef.current.std);
        featuresTensor.dispose();
        return normalizedFeatures as tf.Tensor2D;
      }
      
      return featuresTensor;
    } catch (error) {
      console.error("Preprocessing error:", error);
      return null;
    }
  };

  // Use TensorFlow model for classification
  const classifyFormWithAI = async (landmarks: any[]): Promise<ClassificationResult> => {
    if (!modelRef.current || !scalerRef.current) {
      return { class: 'correct', confidence: 0.5 };
    }
    
    try {
      const inputTensor = preprocessLandmarks(landmarks);
      if (!inputTensor) {
        return { class: 'correct', confidence: 0.5 };
      }
      
      const prediction = modelRef.current.predict(inputTensor) as tf.Tensor;
      const predictionData = await prediction.data();
      
      // Clean up tensors
      inputTensor.dispose();
      prediction.dispose();
      
      // Interpret results (0 = Correct, 1 = Low form)
      const correctConfidence = predictionData[0];
      const lowFormConfidence = predictionData[1];
      
      const predictedClass = correctConfidence > lowFormConfidence ? 'correct' : 'low_form';
      const confidence = Math.max(correctConfidence, lowFormConfidence);
      
      return {
        class: predictedClass,
        confidence: confidence
      };
    } catch (error) {
      console.error("AI classification error:", error);
      return { class: 'correct', confidence: 0.5 };
    }
  };

  // Process landmarks for bicep curl counting and AI classification
  const processLandmarks = async (landmarks: any[]) => {
    setIsProcessing(true);
    
    try {
      const warnings: string[] = [];
      
      // AI-powered form classification
      const aiResult = await classifyFormWithAI(landmarks);
      setClassification(aiResult);
      
      // Get landmark positions
      const leftShoulder = landmarks[POSE_LANDMARKS.LEFT_SHOULDER];
      const leftElbow = landmarks[POSE_LANDMARKS.LEFT_ELBOW];
      const leftWrist = landmarks[POSE_LANDMARKS.LEFT_WRIST];
      
      const rightShoulder = landmarks[POSE_LANDMARKS.RIGHT_SHOULDER];
      const rightElbow = landmarks[POSE_LANDMARKS.RIGHT_ELBOW];
      const rightWrist = landmarks[POSE_LANDMARKS.RIGHT_WRIST];
      
      // Left arm analysis
      if (leftShoulder && leftElbow && leftWrist && 
          leftShoulder.visibility > VISIBILITY_THRESHOLD && 
          leftElbow.visibility > VISIBILITY_THRESHOLD && 
          leftWrist.visibility > VISIBILITY_THRESHOLD) {
        
        const leftArmAnalysis = analyzeArm(
          leftShoulder, leftElbow, leftWrist, 
          'LEFT', aiResult
        );
        
        setLeftAngle(leftArmAnalysis.angle);
        leftStageRef.current = leftArmAnalysis.stage;
        
        if (leftArmAnalysis.counter !== leftCounterRef.current) {
          leftCounterRef.current = leftArmAnalysis.counter;
          setLeftCounter(leftArmAnalysis.counter);
        }
        
        warnings.push(...leftArmAnalysis.warnings.map(w => `Left arm: ${w}`));
      }
      
      // Right arm analysis
      if (rightShoulder && rightElbow && rightWrist && 
          rightShoulder.visibility > VISIBILITY_THRESHOLD && 
          rightElbow.visibility > VISIBILITY_THRESHOLD && 
          rightWrist.visibility > VISIBILITY_THRESHOLD) {
        
        const rightArmAnalysis = analyzeArm(
          rightShoulder, rightElbow, rightWrist, 
          'RIGHT', aiResult
        );
        
        setRightAngle(rightArmAnalysis.angle);
        rightStageRef.current = rightArmAnalysis.stage;
        
        if (rightArmAnalysis.counter !== rightCounterRef.current) {
          rightCounterRef.current = rightArmAnalysis.counter;
          setRightCounter(rightArmAnalysis.counter);
        }
        
        warnings.push(...rightArmAnalysis.warnings.map(w => `Right arm: ${w}`));
      }
      
      // Add AI-based warnings
      if (aiResult.class === 'low_form' && aiResult.confidence > 0.7) {
        warnings.push("AI detected poor form - focus on controlled movement");
      }
      
      setCurrentWarnings(warnings);
      
    } catch (error) {
      console.error("Processing error:", error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      setDebugInfo(`Processing error: ${errorMessage}`);
    } finally {
      setIsProcessing(false);
    }
  };

  // Analyze a single arm for bicep curl counting and form
  const analyzeArm = (
    shoulder: any, elbow: any, wrist: any,
    side: 'LEFT' | 'RIGHT',
    aiResult: ClassificationResult
  ): ArmAnalysis => {
    // Calculate angles
    const curlAngle = calculateAngle(
      [shoulder.x, shoulder.y],
      [elbow.x, elbow.y],
      [wrist.x, wrist.y]
    );
    
    const upperArmAngle = calculateAngle(
      [elbow.x, elbow.y],
      [shoulder.x, shoulder.y],
      [shoulder.x, 1] // Vertical reference
    );
    
    // Update counter
    let counter = (side === 'LEFT') ? leftCounterRef.current : rightCounterRef.current;
    let stage = (side === 'LEFT') ? leftStageRef.current : rightStageRef.current;
    
    if (curlAngle > STAGE_DOWN_THRESHOLD) {
      stage = 'down';
    } else if (curlAngle < STAGE_UP_THRESHOLD && stage === 'down') {
      stage = 'up';
      counter++;
    }
    
    // Check for form issues
    const warnings: string[] = [];
    
    // Loose upper arm
    if (upperArmAngle > LOOSE_UPPER_ARM_THRESHOLD) {
      warnings.push("LOOSE UPPER ARM");
    }
    
    // Poor contraction
    if (aiResult.class === 'low_form' && stage === 'up' && curlAngle > PEAK_CONTRACTION_THRESHOLD) {
      warnings.push("POOR CONTRACTION");
    }
    
    return {
      counter,
      warnings,
      angle: Math.round(curlAngle),
      stage
    };
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
      setDebugInfo("Starting pose detection with AI classification...");
      classifyFrame();
    } else {
      setStatus("Stopped");
      setDebugInfo("AI pose detection stopped");
      if (animationIdRef.current) {
        cancelAnimationFrame(animationIdRef.current);
      }
    }
  };

  // Reset counters
  const resetCounters = () => {
    setLeftCounter(0);
    setRightCounter(0);
    leftCounterRef.current = 0;
    rightCounterRef.current = 0;
    leftStageRef.current = 'down';
    rightStageRef.current = 'down';
    setCurrentWarnings([]);
    setClassification({ class: 'correct', confidence: 0 });
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
          <div className="bg-blue-500 bg-opacity-80 text-white px-3 py-1 rounded-lg text-sm">
            L: {leftAngle}° | {leftStageRef.current}
          </div>
          <div className="bg-purple-500 bg-opacity-80 text-white px-3 py-1 rounded-lg text-sm">
            R: {rightAngle}° | {rightStageRef.current}
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
          onClick={resetCounters}
          className="px-6 py-3 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg transition-colors font-medium flex items-center justify-center gap-2"
        >
          <RefreshCw className="w-4 h-4" />
          Reset
        </button>
      </div>

      {/* Counter Display */}
      <div className="absolute bottom-20 left-4 bg-black bg-opacity-50 text-white px-4 py-2 rounded-lg">
        <div className="flex items-center gap-6">
          <div className="text-center">
            <div className="text-xl font-bold">{leftCounter}</div>
            <div className="text-xs opacity-90">Left</div>
          </div>
          <div className="text-center">
            <div className="text-xl font-bold">{rightCounter}</div>
            <div className="text-xs opacity-90">Right</div>
          </div>
        </div>
      </div>

      {/* Form Feedback */}
      <div className="absolute top-20 left-4 max-w-xs">
        {currentWarnings.length > 0 && (
          <div className="space-y-2">
            {currentWarnings.map((warning, index) => (
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

export default BicepClassifier;