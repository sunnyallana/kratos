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
  youtubeId: string;
  instructions: {
    title: string;
    tips: string[];
    targetMuscles: string[];
    difficulty: string;
    duration: string;
  };
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
      gif: '/assets/exercises-gifs/bicep-curls.gif',
      color: 'from-[#2565EB] to-[#6465F1]',
      youtubeId: 'ykJmrZ5v0Oo',
      instructions: {
        title: 'Perfect Your Bicep Curl',
        tips: [
          'Keep your elbows close to your torso throughout the movement',
          'Start with arms fully extended and weights at your sides',
          'Curl the weights up by contracting your biceps, not swinging',
          'Squeeze at the top and slowly lower with control',
          'Keep your wrists straight and core engaged'
        ],
        targetMuscles: ['Biceps', 'Forearms', 'Core'],
        difficulty: 'Beginner',
        duration: '3-4 sets of 8-12 reps'
      }
    },
    {
      id: 'plank',
      name: 'Plank',
      description: 'Strengthen your core with perfect plank form',
      icon: <Activity className="w-5 h-5" />,
      component: <PlankClassifier />,
      gif: 'assets/exercises-gifs/plank.gif',
      color: 'from-[#9234EA] to-[#6465F1]',
      youtubeId: 'ASdvN_XEl_c',
      instructions: {
        title: 'Master the Perfect Plank',
        tips: [
          'Form a straight line from head to heels - no sagging or piking',
          'Keep your core tight and glutes engaged throughout',
          'Place forearms flat on the ground, elbows under shoulders',
          'Look down at the floor to maintain neutral neck position',
          'Breathe steadily - don\'t hold your breath'
        ],
        targetMuscles: ['Core', 'Shoulders', 'Glutes', 'Back'],
        difficulty: 'Intermediate',
        duration: 'Hold for 30-60 seconds'
      }
    },
    {
      id: 'squat',
      name: 'Squat',
      description: 'Perfect your squat technique with real-time feedback',
      icon: <Activity className="w-5 h-5" />,
      component: <SquatClassifier />,
      gif: 'assets/exercises-gifs/squat.gif',
      color: 'from-[#06976A] to-[#34D399]',
      youtubeId: 'YaXPRqUwItQ',
      instructions: {
        title: 'Execute the Perfect Squat',
        tips: [
          'Stand with feet shoulder-width apart, toes slightly outward',
          'Lower by pushing hips back and bending knees, like sitting in a chair',
          'Keep your chest up and weight on your heels',
          'Descend until thighs are parallel to the floor',
          'Drive through heels to return to starting position'
        ],
        targetMuscles: ['Quadriceps', 'Glutes', 'Hamstrings', 'Core'],
        difficulty: 'Beginner',
        duration: '3-4 sets of 10-15 reps'
      }
    },
    {
      id: 'lunge',
      name: 'Lunge',
      description: 'Improve your lunge form with AI guidance',
      icon: <Activity className="w-5 h-5" />,
      component: <LungeClassifier />,
      gif: 'assets/exercises-gifs/lunges.gif',
      color: 'from-[#22D2EE] to-[#2565EB]',
      youtubeId: 'ASdqJoDPMHA',
      instructions: {
        title: 'Master the Forward Lunge',
        tips: [
          'Step forward with one leg, lowering your hips until both knees are 90°',
          'Keep your front knee directly above your ankle, not pushed out',
          'Keep your back straight and core engaged throughout',
          'Push back to starting position using your front heel',
          'Alternate legs or complete all reps on one side first'
        ],
        targetMuscles: ['Quadriceps', 'Glutes', 'Hamstrings', 'Calves'],
        difficulty: 'Intermediate',
        duration: '3 sets of 10-12 reps per leg'
      }
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
    <div className="min-h-screen bg-gradient-to-br from-black via-[#9234EA] to-black">
      <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPGcgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KPGcgZmlsbD0iIzllYTNiYSIgZmlsbC1vcGFjaXR5PSIwLjAzIj4KPGNpcmNsZSBjeD0iMjkiIGN5PSIyOSIgcj0iMS41Ii8+CjxwYXRoIGQ9Im0yOS01LTUtNWgxMFptMCAxMGwtNS01aDE0bC00IDR6bTAgMTBsLTUtNWgxNGwtNCA0em0wIDEwbC01LTVoMTRsLTQgNHptMCAxMGwtNS01aDE0bC00IDR6bTIwIDEwbC01LTVoMTRsLTQgNHoiLz4KPC9nPgo8L2c+Cjwvc3ZnPg==')] opacity-20"></div>
      
      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Hero Section */}
        <div className="text-center mb-20">
          <div className="flex justify-center mb-6">
            <div className="p-4 bg-gradient-to-r from-[#2565EB] to-[#9234EA] rounded-full shadow-lg">
              <Dumbbell className="w-12 h-12 text-white" />
            </div>
          </div>
          <h1 className="text-6xl font-black bg-gradient-to-r from-white via-[#22D2EE] to-[#9234EA] bg-clip-text text-transparent mb-6">
            AI Fitness Trainer
          </h1>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto leading-relaxed">
            Transform your workouts with cutting-edge AI technology. Get real-time form correction, 
            personalized feedback, and take your fitness to the next level.
          </p>
          <div className="mt-8 flex flex-col sm:flex-row gap-4 justify-center">
            <div className="flex items-center gap-2 text-[#34D399] font-medium">
              <div className="w-2 h-2 bg-[#34D399] rounded-full animate-pulse"></div>
              AI-Powered Analysis
            </div>
            <div className="flex items-center gap-2 text-[#22D2EE] font-medium">
              <div className="w-2 h-2 bg-[#22D2EE] rounded-full animate-pulse"></div>
              Real-time Feedback
            </div>
            <div className="flex items-center gap-2 text-[#9234EA] font-medium">
              <div className="w-2 h-2 bg-[#9234EA] rounded-full animate-pulse"></div>
              Form Correction
            </div>
          </div>
        </div>

        {/* Exercises Grid */}
        <div className="mb-20">
          <h2 className="text-3xl font-bold text-white mb-8 flex items-center gap-3">
            <div className="p-2 bg-gradient-to-r from-[#2565EB] to-[#9234EA] rounded-lg">
              <Activity className="w-6 h-6 text-white" />
            </div>
            Choose Your Exercise
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-2 gap-8">
            {exercises.map((exercise) => (
              <div 
                key={exercise.id}
                onClick={() => setSelectedExercise(exercise)}
                className="group bg-white/10 backdrop-blur-lg rounded-3xl p-8 border border-white/20 hover:border-white/40 transition-all duration-500 cursor-pointer hover:scale-105 hover:shadow-2xl hover:shadow-[#9234EA]/20"
              >
                <div className="flex items-center justify-between mb-6">
                  <div className="flex items-center gap-4">
                    <div className={`p-4 rounded-2xl bg-gradient-to-br ${exercise.color} text-white shadow-lg`}>
                      {exercise.icon}
                    </div>
                    <div>
                      <h3 className="text-2xl font-bold text-white">{exercise.name}</h3>
                      <p className="text-gray-300 text-sm">AI-Guided Training</p>
                    </div>
                  </div>
                  <div className="w-24 h-24 bg-gradient-to-br from-white/20 to-white/10 rounded-2xl overflow-hidden border border-white/20 group-hover:border-white/40 transition-all duration-300">
                    <img 
                      src={exercise.gif} 
                      alt={exercise.name} 
                      className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500"
                    />
                  </div>
                </div>
                <p className="text-gray-300 mb-6 text-lg">{exercise.description}</p>
                <div className="flex justify-between items-center">
                  <span className="text-white font-semibold flex items-center gap-2 group-hover:text-[#22D2EE] transition-colors">
                    Start Training <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                  </span>
                  <div className="flex items-center gap-1 text-gray-400">
                    <div className="w-2 h-2 bg-[#34D399] rounded-full animate-pulse"></div>
                    <span className="text-sm">Ready</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* How It Works Section */}
        <div className="bg-gradient-to-r from-white/10 to-white/5 backdrop-blur-lg rounded-3xl p-10 border border-white/20 shadow-2xl">
          <div className="max-w-5xl mx-auto">
            <h3 className="text-3xl font-bold text-white mb-10 flex items-center gap-4">
              <div className="p-3 bg-gradient-to-r from-[#2565EB] to-[#9234EA] rounded-2xl">
                <Activity className="w-8 h-8 text-white" />
              </div>
              How It Works
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
              {[
                {
                  step: 1,
                  title: "Select Exercise",
                  description: "Choose from our AI-powered exercises tailored to your fitness goals",
                  icon: "🎯"
                },
                {
                  step: 2,
                  title: "Allow Camera",
                  description: "Grant camera permissions for real-time pose detection and analysis",
                  icon: "📹"
                },
                {
                  step: 3,
                  title: "Get Feedback",
                  description: "Our AI analyzes your form and provides instant corrections",
                  icon: "🤖"
                },
                {
                  step: 4,
                  title: "Improve & Track",
                  description: "Follow guidance to perfect your form and track your progress",
                  icon: "📈"
                }
              ].map((item, index) => (
                <div key={item.step} className="relative">
                  <div className="text-center">
                    <div className="mb-4 flex justify-center">
                      <div className="relative">
                        <div className="w-16 h-16 bg-gradient-to-br from-[#2565EB] to-[#9234EA] rounded-full flex items-center justify-center text-white font-bold text-xl shadow-lg">
                          {item.step}
                        </div>
                        <div className="absolute -top-1 -right-1 text-2xl">{item.icon}</div>
                      </div>
                    </div>
                    <h4 className="font-bold text-white text-lg mb-2">{item.title}</h4>
                    <p className="text-gray-300 text-sm leading-relaxed">{item.description}</p>
                  </div>
                  {index < 3 && (
                    <div className="hidden lg:block absolute top-8 -right-4 w-8 h-0.5 bg-gradient-to-r from-[#2565EB] to-[#9234EA]"></div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Footer CTA */}
        <div className="mt-20 text-center">
          <div className="max-w-2xl mx-auto">
            <h3 className="text-2xl font-bold text-white mb-4">
              Ready to Transform Your Fitness Journey?
            </h3>
            <p className="text-gray-300 mb-8 text-lg leading-relaxed">
              Join thousands of users who have improved their form and achieved their fitness goals with our AI-powered personal trainer.
            </p>
            <div className="flex justify-center gap-8 text-sm">
              <div className="flex items-center gap-2 text-[#34D399]">
                <div className="w-2 h-2 bg-[#34D399] rounded-full"></div>
                <span>Free to use</span>
              </div>
              <div className="flex items-center gap-2 text-[#22D2EE]">
                <div className="w-2 h-2 bg-[#22D2EE] rounded-full"></div>
                <span>No signup required</span>
              </div>
              <div className="flex items-center gap-2 text-[#9234EA]">
                <div className="w-2 h-2 bg-[#9234EA] rounded-full"></div>
                <span>Privacy focused</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;