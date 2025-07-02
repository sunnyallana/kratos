import { useState } from 'react';
import { Dumbbell, Activity, Home, User } from 'lucide-react';
import BicepClassifier from './BicepClassifier';
import ExerciseLayout from './ExerciseLayout';
import PlankClassifier from './PlankClassifier';
import SquatClassifier from './SquatClassifier';

type Exercise = {
  id: string;
  name: string;
  description: string;
  icon: JSX.Element;
  component: JSX.Element;
  gif: string;
};

const App = () => {
  const [selectedExercise, setSelectedExercise] = useState<Exercise | null>(null);
  
  const exercises: Exercise[] = [
    {
      id: 'bicep-curl',
      name: 'Bicep Curl',
      description: 'Build arm strength with proper bicep curl form',
      icon: <Dumbbell className="w-5 h-5" />,
      component: <BicepClassifier />,
      gif: '/gifs/bicep-curl.gif'
    },
    {
    id: 'plank',
    name: 'Plank',
    description: 'Strengthen your core with perfect plank form',
    icon: <Activity className="w-5 h-5" />,
    component: <PlankClassifier />,
    gif: '/gifs/plank.gif'
  },
    {
    id: 'squat',
    name: 'Squat',
    description: 'Perfect your squat technique',
    icon: <Activity className="w-5 h-5" />,
    component: <SquatClassifier />,
    gif: '/gifs/squat.gif'
  },
  ];

  if (selectedExercise) {
    return (
      <ExerciseLayout 
        exercise={selectedExercise} 
        onBack={() => setSelectedExercise(null)}
      />
    );
  }

  return (
    <div className="max-w-7xl mx-auto p-6">
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-2">
          AI Fitness Trainer
        </h1>
        <p className="text-gray-600 text-lg">Select an exercise to get started</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {exercises.map((exercise) => (
          <div 
            key={exercise.id}
            onClick={() => setSelectedExercise(exercise)}
            className="bg-white rounded-xl p-6 shadow-lg hover:shadow-xl transition-shadow cursor-pointer border border-gray-100 hover:border-blue-200"
          >
            <div className="flex items-center gap-3 mb-4">
              <div className="bg-blue-100 p-2 rounded-full text-blue-600">
                {exercise.icon}
              </div>
              <h2 className="text-xl font-bold text-gray-800">{exercise.name}</h2>
            </div>
            <p className="text-gray-600 mb-4">{exercise.description}</p>
            <div className="flex justify-between items-center">
              <span className="text-sm font-medium text-blue-600">Start Training</span>
              <div className="w-16 h-16 bg-gray-100 rounded-lg overflow-hidden">
                <img 
                  src={exercise.gif} 
                  alt={exercise.name} 
                  className="w-full h-full object-cover"
                />
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="mt-12 bg-white rounded-xl p-6 shadow-lg">
        <h3 className="font-bold text-xl mb-4 text-gray-800 flex items-center gap-2">
          <Activity className="w-5 h-5" />
          How It Works
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 text-sm">
          <div className="flex items-start gap-3">
            <div className="bg-blue-500 text-white rounded-full w-6 h-6 flex items-center justify-center font-bold text-xs">1</div>
            <div>
              <div className="font-semibold text-gray-800">Select Exercise</div>
              <div className="text-gray-600">Choose from our AI-powered exercises</div>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <div className="bg-blue-500 text-white rounded-full w-6 h-6 flex items-center justify-center font-bold text-xs">2</div>
            <div>
              <div className="font-semibold text-gray-800">Allow Camera</div>
              <div className="text-gray-600">Grant permissions when prompted</div>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <div className="bg-blue-500 text-white rounded-full w-6 h-6 flex items-center justify-center font-bold text-xs">3</div>
            <div>
              <div className="font-semibold text-gray-800">Get Feedback</div>
              <div className="text-gray-600">AI will analyze your form in real-time</div>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <div className="bg-blue-500 text-white rounded-full w-6 h-6 flex items-center justify-center font-bold text-xs">4</div>
            <div>
              <div className="font-semibold text-gray-800">Improve</div>
              <div className="text-gray-600">Follow guidance to perfect your technique</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;