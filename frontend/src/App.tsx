import { useState, useRef, useEffect, useMemo, useCallback } from 'react';
import { Video, VideoOff, AlertCircle } from 'lucide-react';
import BicepCurlDetector from './components/BicepCurlDetector';
import ErrorBoundary from './components/ErrorBoundary';

interface Exercise {
  name: string;
  description: string;
  duration: number;
  repetitions?: number;
  difficulty: string;
}

const App = () => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [selectedExercise, setSelectedExercise] = useState<Exercise | null>(null);
  const [exerciseInProgress, setExerciseInProgress] = useState(false);
  const [countdown, setCountdown] = useState(0);
  const [exerciseWarning, setExerciseWarning] = useState<string | null>(null);

  // Memoize exercises to prevent unnecessary re-renders
  const exercises = useMemo(() => [
    {
      name: "Plank",
      description: "Maintain a push-up position with your body straight",
      duration: 60,
      difficulty: "Intermediate"
    },
    {
      name: "Lunges",
      description: "Step forward and lower your hips until both knees are bent at 90 degrees",
      duration: 45,
      repetitions: 12,
      difficulty: "Beginner"
    },
    {
      name: "Squats",
      description: "Lower your body by bending knees and hips, then return to standing",
      duration: 45,
      repetitions: 15,
      difficulty: "Beginner"
    },
    {
      name: "Bicep Curls",
      description: "Curl weights upward while keeping elbows close to your body",
      duration: 40,
      repetitions: 20,
      difficulty: "Beginner"
    }
  ], []);

  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: 1280,
          height: 720,
          facingMode: 'user' 
        },
        audio: false 
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsCameraOn(true);
      }
    } catch (err) {
      setError('Could not access camera. Please ensure permissions are granted and the device has a camera.');
      console.error('Camera error:', err);
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setIsCameraOn(false);
    }
  }, []);

  const startExercise = useCallback((exercise: Exercise) => {
    setSelectedExercise(exercise);
    setExerciseInProgress(true);
    setCountdown(exercise.duration);
    setExerciseWarning(null);
    
    const timer = setInterval(() => {
      setCountdown(prev => {
        if (prev <= 1) {
          clearInterval(timer);
          setExerciseInProgress(false);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  const handleExerciseWarning = useCallback((message: string) => {
    setExerciseWarning(message);
  }, []);

  // Memoize the detector component to prevent unnecessary remounts
  const BicepDetectorContainer = useMemo(() => {
    return isCameraOn && selectedExercise?.name === "Bicep Curls" ? (
      <ErrorBoundary>
        <BicepCurlDetector 
          key={`detector-${selectedExercise.name}`}
          videoRef={videoRef} 
          onWarning={handleExerciseWarning}
          isExerciseActive={exerciseInProgress && selectedExercise?.name === "Bicep Curls"}
        />
      </ErrorBoundary>
    ) : null;
  }, [isCameraOn, selectedExercise?.name, handleExerciseWarning]);

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-6xl mx-auto">
        <div className="flex justify-between items-center mb-8 bg-white p-4 rounded-lg shadow-sm">
          <h1 className="text-2xl font-bold text-gray-800">Fitness Trainer</h1>
          <div className="flex gap-4">
            {!isCameraOn ? (
              <button
                onClick={startCamera}
                className="flex items-center gap-2 px-4 py-2 rounded-lg bg-blue-500 hover:bg-blue-600 text-white"
              >
                <Video className="w-5 h-5" />
                <span>Start Camera</span>
              </button>
            ) : (
              <button
                onClick={stopCamera}
                className="flex items-center gap-2 px-4 py-2 rounded-lg bg-red-500 hover:bg-red-600 text-white"
              >
                <VideoOff className="w-5 h-5" />
                <span>Stop Camera</span>
              </button>
            )}
          </div>
        </div>

        {error && (
          <div className="mb-6 p-4 rounded-lg bg-red-100 text-red-800">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-5 h-5" />
              <span>{error}</span>
            </div>
          </div>
        )}

        {exerciseWarning && (
          <div className="mb-6 p-4 rounded-lg bg-red-100 text-red-800">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-5 h-5" />
              <span>{exerciseWarning}</span>
            </div>
          </div>
        )}

        <div className="flex flex-col lg:flex-row gap-6">
          {/* Left Column - Camera Feed */}
          <div className="lg:w-2/3">
            <div className="bg-white rounded-xl shadow-sm overflow-hidden">
              <div className="bg-gray-800 flex justify-center items-center p-4">
                <video 
                  ref={videoRef} 
                  autoPlay 
                  playsInline 
                  muted 
                  className="w-full max-w-[600px] rounded-lg"
                />
              </div>
            </div>

            {BicepDetectorContainer}

            <div className="mt-6 bg-white rounded-xl shadow-sm p-6">
              <h2 className="text-xl font-semibold text-gray-800 mb-4">Camera Instructions</h2>
              <ul className="space-y-3 text-sm text-gray-700">
                <li className="flex items-start gap-2">
                  <span className="text-blue-500">•</span>
                  <span>Stand 2-3 meters from the camera</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-blue-500">•</span>
                  <span>Ensure your entire body is visible</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-blue-500">•</span>
                  <span>Good lighting improves accuracy</span>
                </li>
              </ul>
            </div>
          </div>

          {/* Right Column - Exercises */}
          <div className="lg:w-1/3">
            <div className="bg-white rounded-xl shadow-sm p-6">
              <h2 className="text-xl font-semibold text-gray-800 mb-4">Available Exercises</h2>
              <div className="space-y-4">
                {exercises.map((exercise) => (
                  <div 
                    key={exercise.name}
                    className={`border rounded-lg p-4 transition-all cursor-pointer ${
                      selectedExercise?.name === exercise.name && exerciseInProgress
                        ? 'border-blue-500 bg-blue-50'
                        : 'border-gray-200 hover:border-blue-300 hover:bg-blue-50'
                    }`}
                    onClick={() => !exerciseInProgress && startExercise(exercise)}
                  >
                    <div className="flex justify-between items-start">
                      <h3 className="font-medium text-gray-800">{exercise.name}</h3>
                      <span className="text-xs px-2 py-1 bg-gray-100 text-gray-600 rounded">
                        {exercise.difficulty}
                      </span>
                    </div>
                    <p className="text-sm text-gray-600 mt-1">{exercise.description}</p>
                    <div className="flex justify-between items-center mt-3 text-sm">
                      <span className="text-gray-500">
                        {exercise.repetitions ? `${exercise.repetitions} reps` : `${exercise.duration} sec`}
                      </span>
                      <button 
                        className="px-3 py-1 bg-blue-500 text-white rounded text-xs hover:bg-blue-600 disabled:opacity-50"
                        disabled={exerciseInProgress || !isCameraOn}
                      >
                        Start
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {exerciseInProgress && selectedExercise && (
              <div className="mt-6 bg-white rounded-xl shadow-sm p-6">
                <h2 className="text-xl font-semibold text-gray-800 mb-4">Current Exercise</h2>
                <div className="flex justify-between items-center">
                  <span className="font-medium">{selectedExercise.name}</span>
                  <span className="text-blue-600">{countdown}s remaining</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2.5 mt-4">
                  <div 
                    className="bg-blue-600 h-2.5 rounded-full" 
                    style={{ width: `${(countdown / selectedExercise.duration) * 100}%` }}
                  ></div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;