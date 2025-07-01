import { useEffect, useRef, useState, useCallback } from 'react';
import { AlertCircle, Loader2 } from 'lucide-react';
import * as tf from '@tensorflow/tfjs';

interface BicepCurlDetectorProps {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  onWarning: (message: string) => void;
  isExerciseActive: boolean;
}

// Declare MediaPipe globals (loaded via CDN)
declare global {
  interface Window {
    Pose: any;
    drawConnectors: any;
    drawLandmarks: any;
    POSE_CONNECTIONS: any;
    POSE_LANDMARKS: any;
  }
}

const BicepCurlDetector = ({ videoRef, onWarning, isExerciseActive }: BicepCurlDetectorProps) => {
  const [debugLogs, setDebugLogs] = useState<string[]>([]);
  const [loadingProgress, setLoadingProgress] = useState<string>('');
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [leftCounter, setLeftCounter] = useState(0);
  const [rightCounter, setRightCounter] = useState(0);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [classification, setClassification] = useState<string>('-');
  const [confidence, setConfidence] = useState<string>('-');
  
  const poseRef = useRef<any>(null);
  const modelRef = useRef<tf.LayersModel | null>(null);
  const scalerRef = useRef<{ mean: tf.Tensor; std: tf.Tensor } | null>(null);
  const leftStageRef = useRef<'up' | 'down'>('down');
  const rightStageRef = useRef<'up' | 'down'>('down');
  const animationRef = useRef<number | null>(null);

  // Thresholds
  const STAGE_UP_THRESHOLD = 90;
  const STAGE_DOWN_THRESHOLD = 120;
  const LOOSE_UPPER_ARM_THRESHOLD = 40;
  const PEAK_CONTRACTION_THRESHOLD = 60;
  const VISIBILITY_THRESHOLD = 0.65;

  const addDebugLog = (message: string) => {
    const timestamp = new Date().toISOString().substring(11, 23);
    const logMessage = `[${timestamp}] ${message}`;
    console.log(logMessage);
    setDebugLogs(prev => [...prev.slice(-9), logMessage]);
  };

  // Load MediaPipe scripts dynamically
  const loadMediaPipeScripts = (): Promise<void> => {
    return new Promise((resolve, reject) => {
      if (window.Pose) {
        resolve();
        return;
      }

      const scripts = [
        'https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js',
        'https://cdn.jsdelivr.net/npm/@mediapipe/control_utils/control_utils.js',
        'https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js',
        'https://cdn.jsdelivr.net/npm/@mediapipe/pose/pose.js'
      ];

      let loadedCount = 0;
      
      scripts.forEach((src) => {
        const script = document.createElement('script');
        script.src = src;
        script.crossOrigin = 'anonymous';
        script.onload = () => {
          loadedCount++;
          if (loadedCount === scripts.length) {
            addDebugLog('All MediaPipe scripts loaded');
            resolve();
          }
        };
        script.onerror = () => reject(new Error(`Failed to load ${src}`));
        document.head.appendChild(script);
      });
    });
  };

  useEffect(() => {
    let isMounted = true;

    const initialize = async () => {
      try {
        addDebugLog('Initializing detector');
        
        // Load MediaPipe scripts first
        setLoadingProgress('Loading MediaPipe...');
        await loadMediaPipeScripts();
        
        if (!isMounted) return;

        // Initialize MediaPipe Pose
        const pose = new window.Pose({
          locateFile: (file: string) => {
            addDebugLog(`Loading MediaPipe file: ${file}`);
            return `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`;
          },
        });

        pose.setOptions({
          modelComplexity: 1,
          smoothLandmarks: true,
          enableSegmentation: false,
          smoothSegmentation: true,
          minDetectionConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });

        pose.onResults((results: any) => {
          if (!isMounted) return;
          try {
            onPoseResults(results);
          } catch (error) {
            addDebugLog(`Pose results error: ${error}`);
          }
        });

        poseRef.current = pose;
        addDebugLog('MediaPipe Pose initialized');

        // Load TensorFlow model and scaler
        setLoadingProgress('Loading AI model...');
        addDebugLog('Starting model loading');
        
        try {
          // Try different model paths
          let model: tf.LayersModel;
          let scalerData: any;
          
          try {
            // Try absolute path first
            model = await tf.loadLayersModel('/model/bicep_classifier/model.json');
            const scalerResponse = await fetch('/model/input_scaler.json');
            scalerData = await scalerResponse.json();
          } catch (error) {
            // Try relative path
            model = await tf.loadLayersModel('./model/bicep_classifier/model.json');
            const scalerResponse = await fetch('./model/input_scaler.json');
            scalerData = await scalerResponse.json();
          }
          
          if (!isMounted) {
            addDebugLog('Component unmounted during model load');
            model.dispose();
            return;
          }

          modelRef.current = model;
          scalerRef.current = {
            mean: tf.tensor(scalerData.mean),
            std: tf.tensor(scalerData.std),
          };

          setIsModelLoaded(true);
          setLoadingProgress('');
          addDebugLog('Model initialized successfully');
        } catch (modelError) {
          addDebugLog(`Model loading failed: ${modelError}`);
          setLoadingProgress('Model loading failed - continuing with pose detection only');
          // Continue without model for pose detection
        }
        
      } catch (error) {
        if (isMounted) {
          const errMsg = `Initialization failed: ${error}`;
          console.error(errMsg);
          addDebugLog(errMsg);
          setLoadingProgress('Initialization failed');
          onWarning('Failed to initialize detector. Please refresh.');
        }
      }
    };

    initialize();

    return () => {
      isMounted = false;
      addDebugLog('Starting cleanup');
      
      // Cleanup MediaPipe
      if (poseRef.current) {
        try {
          poseRef.current.close();
          addDebugLog('MediaPipe closed');
        } catch (error) {
          addDebugLog(`MediaPipe cleanup error: ${error}`);
        }
      }

      // Cleanup TensorFlow
      tf.tidy(() => {
        if (modelRef.current) {
          modelRef.current.dispose();
          addDebugLog('TF model disposed');
        }
        if (scalerRef.current) {
          scalerRef.current.mean.dispose();
          scalerRef.current.std.dispose();
          addDebugLog('Scaler tensors disposed');
        }
      });

      // Cleanup animation
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
        addDebugLog('Animation stopped');
      }
      
      addDebugLog('Cleanup completed');
    };
  }, [onWarning]);

  useEffect(() => {
    if (isExerciseActive && poseRef.current) {
      startDetection();
    } else {
      stopDetection();
    }
    
    return () => {
      stopDetection();
    };
  }, [isExerciseActive]);

  const startDetection = useCallback(() => {
    if (!isRunning && poseRef.current) {
      addDebugLog('Starting detection');
      setIsRunning(true);
      classifyFrame();
    }
  }, [isRunning]);

  const stopDetection = () => {
    addDebugLog('Stopping detection');
    setIsRunning(false);
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }
  };

  const onPoseResults = (results: any) => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!ctx || !canvas) {
      addDebugLog('No canvas context available');
      return;
    }

    try {
      ctx.save();
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Draw the video frame
      if (results.image) {
        ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);
      }

      if (results.poseLandmarks) {
        // Draw pose landmarks and connections
        window.drawConnectors(ctx, results.poseLandmarks, window.POSE_CONNECTIONS, {
          color: '#00FF00',
          lineWidth: 2,
        });
        window.drawLandmarks(ctx, results.poseLandmarks, {
          color: '#FF0000',
          lineWidth: 1,
          radius: 2,
        });

        processLandmarks(results.poseLandmarks);
      } else {
        addDebugLog('No pose landmarks detected');
        // Draw "No human detected" message
        ctx.font = '24px Arial';
        ctx.fillStyle = 'red';
        ctx.textAlign = 'center';
        ctx.fillText('NO HUMAN DETECTED', canvas.width / 2, canvas.height / 2);
      }

      ctx.restore();
    } catch (error) {
      addDebugLog(`Drawing error: ${error}`);
    }
  };

  const processLandmarks = (landmarks: any) => {
    if (!modelRef.current || !scalerRef.current) {
      // Just do pose analysis without ML classification
      analyzeBicepCurls(landmarks, 0); // Assume good form
      return;
    }

    try {
      tf.tidy(() => {
        // Extract important landmarks
        const importantLandmarks = [
          'NOSE',
          'LEFT_SHOULDER',
          'RIGHT_SHOULDER',
          'RIGHT_ELBOW',
          'LEFT_ELBOW',
          'RIGHT_WRIST',
          'LEFT_WRIST',
          'LEFT_HIP',
          'RIGHT_HIP',
        ];

        const landmarkData: Record<string, number> = {};
        importantLandmarks.forEach((lm) => {
          const landmark = landmarks[window.POSE_LANDMARKS[lm]];
          if (!landmark) {
            addDebugLog(`Missing landmark: ${lm}`);
            return;
          }
          const key = lm.toLowerCase();
          landmarkData[`${key}_x`] = landmark.x;
          landmarkData[`${key}_y`] = landmark.y;
          landmarkData[`${key}_z`] = landmark.z;
          landmarkData[`${key}_v`] = landmark.visibility;
        });

        // Preprocess landmarks
        const inputData = preprocessLandmarks(landmarkData);

        // Make prediction
        const prediction = modelRef.current!.predict(inputData) as tf.Tensor;
        const predictionData = prediction.dataSync();

        // Interpret results (0 = Correct, 1 = Low form)
        const confidence = Math.max(predictionData[0], predictionData[1]);
        const predictedClass = predictionData[0] > predictionData[1] ? 0 : 1;

        // Update classification display
        const confidencePercent = (confidence * 100).toFixed(1);
        if (predictedClass === 0) {
          setClassification('Good Form');
        } else {
          setClassification('Poor Form');
        }
        setConfidence(`${confidencePercent}%`);

        addDebugLog(`Prediction: ${predictedClass === 0 ? 'Good' : 'Poor'} form (${confidencePercent}% conf)`);

        // Analyze bicep curls for counting and form feedback
        analyzeBicepCurls(landmarks, predictedClass);
      });
    } catch (error) {
      addDebugLog(`Processing error: ${error}`);
    }
  };

  const preprocessLandmarks = (landmarks: Record<string, number>) => {
    const orderedFeatures: number[] = [];

    const BICEP_LANDMARKS = [
      'NOSE',
      'LEFT_SHOULDER',
      'RIGHT_SHOULDER',
      'RIGHT_ELBOW',
      'LEFT_ELBOW',
      'RIGHT_WRIST',
      'LEFT_WRIST',
      'LEFT_HIP',
      'RIGHT_HIP',
    ];

    BICEP_LANDMARKS.forEach((landmark) => {
      const key = landmark.toLowerCase();
      orderedFeatures.push(
        landmarks[`${key}_x`] || 0,
        landmarks[`${key}_y`] || 0,
        landmarks[`${key}_z`] || 0,
        landmarks[`${key}_v`] || 0
      );
    });

    const featuresTensor = tf.tensor2d([orderedFeatures]);
    const normalizedFeatures = featuresTensor
      .sub(scalerRef.current!.mean)
      .div(scalerRef.current!.std);

    return normalizedFeatures;
  };

  const analyzeBicepCurls = (landmarks: any, predictedClass: number) => {
    const newWarnings: string[] = [];

    // Analyze left arm
    const leftAnalysis = analyzeArm(landmarks, 'LEFT', predictedClass);
    if (leftAnalysis) {
      setLeftCounter(leftAnalysis.counter);
      if (leftAnalysis.warnings.length > 0) {
        newWarnings.push(...leftAnalysis.warnings.map((w) => `LEFT ARM: ${w}`));
      }
    }

    // Analyze right arm
    const rightAnalysis = analyzeArm(landmarks, 'RIGHT', predictedClass);
    if (rightAnalysis) {
      setRightCounter(rightAnalysis.counter);
      if (rightAnalysis.warnings.length > 0) {
        newWarnings.push(...rightAnalysis.warnings.map((w) => `RIGHT ARM: ${w}`));
      }
    }

    setWarnings(newWarnings);
    if (newWarnings.length > 0) {
      const warningMessage = newWarnings.join('\n');
      onWarning(warningMessage);
      addDebugLog(`Warning: ${warningMessage}`);
    }
  };

  const analyzeArm = (
    landmarks: any,
    side: 'LEFT' | 'RIGHT',
    predictedClass: number
  ) => {
    const shoulder = landmarks[window.POSE_LANDMARKS[`${side}_SHOULDER`]];
    const elbow = landmarks[window.POSE_LANDMARKS[`${side}_ELBOW`]];
    const wrist = landmarks[window.POSE_LANDMARKS[`${side}_WRIST`]];

    if (!shoulder || !elbow || !wrist) {
      addDebugLog(`Missing landmarks for ${side} arm`);
      return null;
    }

    // Check visibility
    if (
      shoulder.visibility < VISIBILITY_THRESHOLD ||
      elbow.visibility < VISIBILITY_THRESHOLD ||
      wrist.visibility < VISIBILITY_THRESHOLD
    ) {
      addDebugLog(`Low visibility for ${side} arm`);
      return null;
    }

    // Calculate angles
    const curlAngle = calculateAngle(
      [shoulder.x, shoulder.y],
      [elbow.x, elbow.y],
      [wrist.x, wrist.y]
    );

    const upperArmAngle = calculateAngle(
      [elbow.x, elbow.y],
      [shoulder.x, shoulder.y],
      [shoulder.x, 1]
    );

    // Update counter
    let counter = side === 'LEFT' ? leftCounter : rightCounter;
    const stage = side === 'LEFT' ? leftStageRef.current : rightStageRef.current;

    if (curlAngle > STAGE_DOWN_THRESHOLD) {
      if (side === 'LEFT') {
        leftStageRef.current = 'down';
      } else {
        rightStageRef.current = 'down';
      }
    } else if (curlAngle < STAGE_UP_THRESHOLD && stage === 'down') {
      if (side === 'LEFT') {
        leftStageRef.current = 'up';
      } else {
        rightStageRef.current = 'up';
      }
      counter++;
      addDebugLog(`${side} arm rep counted: ${counter}`);
    }

    // Check for form issues
    const warnings: string[] = [];

    // Loose upper arm
    if (upperArmAngle > LOOSE_UPPER_ARM_THRESHOLD) {
      warnings.push('LOOSE UPPER ARM');
      addDebugLog(`${side} arm loose upper arm detected`);
    }

    // Poor contraction
    if (
      predictedClass === 1 &&
      (side === 'LEFT' ? leftStageRef.current : rightStageRef.current) === 'up' &&
      curlAngle > PEAK_CONTRACTION_THRESHOLD
    ) {
      warnings.push('POOR CONTRACTION');
      addDebugLog(`${side} arm poor contraction detected`);
    }

    return {
      counter,
      warnings,
    };
  };

  const calculateAngle = (point1: number[], point2: number[], point3: number[]) => {
    const vector1 = [point1[0] - point2[0], point1[1] - point2[1]];
    const vector2 = [point3[0] - point2[0], point3[1] - point2[1]];

    const dotProduct = vector1[0] * vector2[0] + vector1[1] * vector2[1];
    const magnitude1 = Math.sqrt(vector1[0] ** 2 + vector1[1] ** 2);
    const magnitude2 = Math.sqrt(vector2[0] ** 2 + vector2[1] ** 2);

    const angleRad = Math.acos(dotProduct / (magnitude1 * magnitude2));
    let angleDeg = angleRad * (180 / Math.PI);

    angleDeg = angleDeg > 180 ? 360 - angleDeg : angleDeg;

    return angleDeg;
  };

  const classifyFrame = async () => {
    if (!isRunning || !poseRef.current || !videoRef.current) {
      addDebugLog(`Skipping classification - running: ${isRunning}, pose: ${!!poseRef.current}, video: ${!!videoRef.current}`);
      return;
    }

    try {
      if (!videoRef.current.srcObject) {
        addDebugLog('No video stream available');
        return;
      }

      await poseRef.current.send({ image: videoRef.current });
    } catch (error) {
      addDebugLog(`Classification error: ${error}`);
    }

    if (isRunning) {
      animationRef.current = requestAnimationFrame(classifyFrame);
    }
  };

  // Set canvas size to match video
  useEffect(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (video && canvas) {
      const updateCanvasSize = () => {
        canvas.width = video.clientWidth;
        canvas.height = video.clientHeight;
        addDebugLog(`Canvas resized to ${canvas.width}x${canvas.height}`);
      };

      // Initial size
      updateCanvasSize();

      const resizeObserver = new ResizeObserver(updateCanvasSize);
      resizeObserver.observe(video);
      
      return () => resizeObserver.disconnect();
    }
  }, []);

  return (
    <div className="mt-4">
      <div className="relative">
        <canvas
          ref={canvasRef}
          className="absolute top-0 left-0 w-full h-full pointer-events-none"
          style={{ zIndex: 10 }}
        />
      </div>

      {/* Classification Results */}
      {isModelLoaded && (
        <div className="mt-4 bg-white rounded-lg p-4 shadow">
          <h3 className="text-lg font-semibold mb-2">AI Classification</h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <span className="font-semibold">Form:</span> 
              <span className={`ml-2 ${classification === 'Good Form' ? 'text-green-600' : 'text-red-600'}`}>
                {classification}
              </span>
            </div>
            <div>
              <span className="font-semibold">Confidence:</span> 
              <span className="ml-2">{confidence}</span>
            </div>
          </div>
        </div>
      )}

      {/* Debug panel */}
      <div className="mt-4 bg-gray-100 rounded-lg p-4 text-sm">
        <h4 className="font-bold mb-2">Debug Information</h4>
        <div className="mb-2">
          <span className="font-semibold">Status:</span> {loadingProgress || (poseRef.current ? 'Ready' : 'Loading...')}
          {isModelLoaded && (
            <span className="ml-2 text-green-600">✓ AI Model Loaded</span>
          )}
          {poseRef.current && (
            <span className="ml-2 text-blue-600">✓ Pose Detection Ready</span>
          )}
        </div>
        <div className="max-h-40 overflow-y-auto bg-black text-green-400 p-2 rounded font-mono text-xs">
          {debugLogs.map((log, i) => (
            <div key={i}>{log}</div>
          ))}
        </div>
        <div className="mt-2 grid grid-cols-2 gap-2">
          <div>
            <span className="font-semibold">TF Memory:</span> {tf.memory().numTensors} tensors
          </div>
          <div>
            <span className="font-semibold">Detection:</span> {isRunning ? 'Active' : 'Inactive'}
          </div>
        </div>
      </div>

      <div className="mt-4 bg-white rounded-lg p-4 shadow">
        <h3 className="text-lg font-semibold mb-2">Bicep Curl Counter</h3>
        <div className="flex justify-between mb-4">
          <div className="text-center">
            <p className="text-sm text-gray-600">Left Arm</p>
            <p className="text-2xl font-bold">{leftCounter}</p>
          </div>
          <div className="text-center">
            <p className="text-sm text-gray-600">Right Arm</p>
            <p className="text-2xl font-bold">{rightCounter}</p>
          </div>
        </div>

        {warnings.length > 0 ? (
          <div className="mt-4">
            <h4 className="text-md font-semibold mb-2 text-red-600">Form Warnings</h4>
            <ul className="space-y-2">
              {warnings.map((warning, index) => (
                <li key={index} className="flex items-start text-red-600">
                  <AlertCircle className="w-4 h-4 mt-1 mr-2 flex-shrink-0" />
                  <span>{warning}</span>
                </li>
              ))}
            </ul>
          </div>
        ) : isRunning ? (
          <div className="mt-4 p-3 bg-green-50 rounded-md">
            <p className="text-green-700 font-medium">Good form! Keep it up!</p>
          </div>
        ) : null}

        <div className="mt-4 flex gap-2">
          <button
            onClick={startDetection}
            disabled={isRunning || !poseRef.current}
            className={`px-4 py-2 rounded-md flex items-center ${
              isRunning || !poseRef.current
                ? 'bg-gray-300 cursor-not-allowed'
                : 'bg-blue-500 hover:bg-blue-600 text-white'
            }`}
          >
            {!poseRef.current && (
              <Loader2 className="animate-spin mr-2 h-4 w-4" />
            )}
            {!poseRef.current ? (loadingProgress || 'Loading...') : 'Start Detection'}
          </button>
          <button
            onClick={stopDetection}
            disabled={!isRunning}
            className={`px-4 py-2 rounded-md ${
              !isRunning
                ? 'bg-gray-300 cursor-not-allowed'
                : 'bg-red-500 hover:bg-red-600 text-white'
            }`}
          >
            Stop Detection
          </button>
        </div>
      </div>
    </div>
  );
};

export default BicepCurlDetector;