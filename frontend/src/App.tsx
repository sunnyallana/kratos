import { useState } from 'react';
import { Dumbbell, Activity, Home, User, ArrowRight, ChevronRight } from 'lucide-react';
import BicepClassifier from './components/BicepClassifier';
import ExerciseLayout from './components/ExerciseLayout';
import PlankClassifier from './components/PlankClassifier';
import SquatClassifier from './components/SquatClassifier';
import LungeClassifier from './components/LungeClassifier';

type Exercise = {
  id: string;
  name: string;
  description: string;
  icon: JSX.Element;
  component: JSX.Element;
  gif: string;
  color: string;
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
      gif: '/gifs/bicep-curl.gif',
      color: 'from-blue-400 to-blue-600'
    },
    {
      id: 'plank',
      name: 'Plank',
      description: 'Strengthen your core with perfect plank form',
      icon: <Activity className="w-5 h-5" />,
      component: <PlankClassifier />,
      gif: '/gifs/plank.gif',
      color: 'from-purple-400 to-purple-600'
    },
    {
      id: 'squat',
      name: 'Squat',
      description: 'Perfect your squat technique with real-time feedback',
      icon: <Activity className="w-5 h-5" />,
      component: <SquatClassifier />,
      gif: '/gifs/squat.gif',
      color: 'from-green-400 to-green-600'
    },
    {
      id: 'lunge',
      name: 'Lunge',
      description: 'Improve your lunge form with AI guidance',
      icon: <Activity className="w-5 h-5" />,
      component: <LungeClassifier />,
      gif: '/gifs/lunge.gif',
      color: 'from-orange-400 to-orange-600'
    }
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
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Hero Section */}
        <div className="text-center mb-16">
          <h1 className="text-5xl font-extrabold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-4">
            AI Fitness Trainer
          </h1>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Transform your workouts with AI-powered real-time form correction and personalized feedback
          </p>
        </div>

        {/* Exercises Grid */}
        <div className="mb-20">
          <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-2">
            <Activity className="text-blue-500" />
            Available Exercises
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {exercises.map((exercise) => (
              <div 
                key={exercise.id}
                onClick={() => setSelectedExercise(exercise)}
                className={`group bg-white rounded-2xl p-6 shadow-md hover:shadow-xl transition-all duration-300 cursor-pointer border border-transparent hover:border-${exercise.color.split(' ')[1]} overflow-hidden relative`}
              >
                <div className={`absolute inset-0 bg-gradient-to-br opacity-0 group-hover:opacity-10 transition-opacity duration-300 ${exercise.color}`}></div>
                <div className="flex flex-col h-full">
                  <div className="flex items-center gap-4 mb-4">
                    <div className={`p-3 rounded-full bg-gradient-to-br ${exercise.color} text-white`}>
                      {exercise.icon}
                    </div>
                    <h3 className="text-xl font-bold text-gray-800">{exercise.name}</h3>
                  </div>
                  <p className="text-gray-600 mb-6 flex-grow">{exercise.description}</p>
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium text-blue-600 flex items-center gap-1">
                      Start Training <ArrowRight className="w-4 h-4" />
                    </span>
                    <div className="w-20 h-20 bg-gray-100 rounded-lg overflow-hidden border border-gray-200">
                      <img 
                        src={exercise.gif} 
                        alt={exercise.name} 
                        className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                      />
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* How It Works Section */}
        <div className="bg-white rounded-2xl p-8 shadow-lg">
          <div className="max-w-4xl mx-auto">
            <h3 className="text-2xl font-bold text-gray-800 mb-8 flex items-center gap-3">
              <div className="p-2 bg-blue-100 text-blue-600 rounded-lg">
                <Activity className="w-6 h-6" />
              </div>
              How It Works
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {[
                {
                  step: 1,
                  title: "Select Exercise",
                  description: "Choose from our AI-powered exercises"
                },
                {
                  step: 2,
                  title: "Allow Camera",
                  description: "Grant permissions when prompted"
                },
                {
                  step: 3,
                  title: "Get Feedback",
                  description: "AI analyzes your form in real-time"
                },
                {
                  step: 4,
                  title: "Improve Technique",
                  description: "Follow guidance to perfect your form"
                }
              ].map((item) => (
                <div key={item.step} className="flex items-start gap-4">
                  <div className="flex-shrink-0 bg-gradient-to-br from-blue-500 to-blue-600 text-white rounded-full w-8 h-8 flex items-center justify-center font-bold text-sm">
                    {item.step}
                  </div>
                  <div>
                    <h4 className="font-semibold text-gray-800">{item.title}</h4>
                    <p className="text-gray-600 text-sm mt-1">{item.description}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Footer CTA */}
        <div className="mt-16 text-center">
          <h3 className="text-xl font-medium text-gray-700 mb-4">
            Ready to transform your workouts?
          </h3>
          <p className="text-gray-500 mb-6 max-w-2xl mx-auto">
            Select an exercise above to get started with AI-powered form correction today.
          </p>
        </div>
      </div>
    </div>
  );
};

export default App;