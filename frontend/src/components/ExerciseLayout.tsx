import { ChevronLeft, Camera, Activity, Target, Timer, Zap, Users, TrendingUp, Play } from 'lucide-react';

type ExerciseLayoutProps = {
  exercise: {
    name: string;
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
  onBack: () => void;
};

const ExerciseLayout = ({ exercise, onBack }: ExerciseLayoutProps) => {
  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty.toLowerCase()) {
      case 'beginner':
        return 'text-[#34D399] bg-[#06976A]/20 border-[#06976A]/30';
      case 'intermediate':
        return 'text-[#22D2EE] bg-[#22D2EE]/20 border-[#22D2EE]/30';
      case 'advanced':
        return 'text-[#9234EA] bg-[#9234EA]/20 border-[#9234EA]/30';
      default:
        return 'text-[#2565EB] bg-[#2565EB]/20 border-[#2565EB]/30';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-black via-[#9234EA] to-black">
      <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPGcgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KPGcgZmlsbD0iIzllYTNiYSIgZmlsbC1vcGFjaXR5PSIwLjAzIj4KPGNpcmNsZSBjeD0iMjkiIGN5PSIyOSIgcj0iMS41Ii8+CjxwYXRoIGQ9Im0yOS01LTUtNWgxMFptMCAxMGwtNS01aDE0bC00IDR6bTAgMTBsLTUtNWgxNGwtNCA0em0wIDEwbC01LTVoMTRsLTQgNHptMCAxMGwtNS01aDE0bC00IDR6bTIwIDEwbC01LTVoMTRsLTQgNHoiLz4KPC9nPgo8L2c+Cjwvc3ZnPg==')] opacity-20"></div>
      
      <div className="relative max-w-7xl mx-auto p-6">
        <button
          onClick={onBack}
          className="flex items-center gap-3 text-white hover:text-[#22D2EE] mb-8 transition-colors duration-300 group"
        >
          <div className="p-2 bg-white/10 backdrop-blur-lg rounded-full border border-white/20 group-hover:border-white/40 transition-all duration-300">
            <ChevronLeft className="w-5 h-5 group-hover:-translate-x-1 transition-transform" />
          </div>
          <span className="font-semibold">Back to Exercises</span>
        </button>

        <div className="grid grid-cols-1 xl:grid-cols-4 gap-8">
          {/* Main Training Area */}
          <div className="xl:col-span-3 space-y-6">
            {/* Training Header */}
            <div className="bg-gradient-to-r from-white/10 to-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <div className={`p-3 rounded-xl bg-gradient-to-br ${exercise.color} text-white`}>
                    <Activity className="w-6 h-6" />
                  </div>
                  <div>
                    <h1 className="text-2xl font-bold text-white">{exercise.name} Training</h1>
                    <p className="text-gray-300">AI-Powered Form Analysis</p>
                  </div>
                </div>
                <div className="flex items-center gap-4">
                  <div className="flex items-center gap-2 text-[#34D399]">
                    <div className="w-2 h-2 bg-[#34D399] rounded-full animate-pulse"></div>
                    <span className="text-sm font-medium">AI Active</span>
                  </div>
                  <div className="flex items-center gap-2 text-[#22D2EE]">
                    <Camera className="w-4 h-4" />
                    <span className="text-sm font-medium">Camera Ready</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Camera and Pose Detection */}
            <div className="bg-gradient-to-r from-white/10 to-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
              {exercise.component}
            </div>

            {/* YouTube Video Tutorial - Now Below Camera */}
            <div className="bg-gradient-to-r from-white/10 to-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
              <h3 className="font-bold text-xl mb-4 text-white flex items-center gap-3">
                <div className="p-2 bg-gradient-to-r from-[#2565EB] to-[#9234EA] rounded-lg">
                  <Play className="w-5 h-5 text-white" />
                </div>
                Video Tutorial - {exercise.name}
              </h3>
              <div className="relative aspect-video bg-black rounded-xl overflow-hidden border border-white/20 shadow-lg">
                <iframe
                  src={`https://www.youtube.com/embed/${exercise.youtubeId}?rel=0&modestbranding=1&showinfo=0`}
                  title={`${exercise.name} Tutorial`}
                  className="w-full h-full"
                  frameBorder="0"
                  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                  allowFullScreen
                />
              </div>
              <div className="mt-4 p-4 bg-gradient-to-r from-[#2565EB]/20 to-[#9234EA]/20 rounded-lg border border-[#2565EB]/30">
                <p className="text-gray-300 leading-relaxed">
                  <span className="text-[#22D2EE] font-semibold">Watch first:</span> This professional demonstration shows proper {exercise.name.toLowerCase()} technique. Study the form before starting your AI-guided training session for best results.
                </p>
              </div>
            </div>
          </div>

          {/* Exercise Instructions Sidebar */}
          <div className="xl:col-span-1 space-y-6">
            {/* Exercise Info Card */}
            <div className="bg-gradient-to-r from-white/10 to-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
              <h2 className="text-2xl font-bold text-white mb-4">{exercise.name}</h2>
              <div className="aspect-w-16 aspect-h-9 bg-gradient-to-br from-white/20 to-white/10 rounded-xl overflow-hidden mb-6 border border-white/20">
                <img 
                  src={exercise.gif} 
                  alt={`How to perform ${exercise.name}`}
                  className="w-full h-full object-cover"
                />
              </div>
              
              {/* Exercise Stats */}
              <div className="grid grid-cols-2 gap-4 mb-6">
                <div className="bg-white/10 rounded-lg p-3 text-center border border-white/20">
                  <Target className="w-5 h-5 text-[#22D2EE] mx-auto mb-1" />
                  <div className="text-xs text-gray-300">Difficulty</div>
                  <div className={`text-sm font-semibold px-2 py-1 rounded-full border ${getDifficultyColor(exercise.instructions.difficulty)}`}>
                    {exercise.instructions.difficulty}
                  </div>
                </div>
                <div className="bg-white/10 rounded-lg p-3 text-center border border-white/20">
                  <Timer className="w-5 h-5 text-[#34D399] mx-auto mb-1" />
                  <div className="text-xs text-gray-300">Duration</div>
                  <div className="text-sm font-semibold text-white">{exercise.instructions.duration}</div>
                </div>
              </div>

              {/* Target Muscles */}
              <div className="mb-6">
                <h4 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                  <Zap className="w-4 h-4 text-[#6465F1]" />
                  Target Muscles
                </h4>
                <div className="flex flex-wrap gap-2">
                  {exercise.instructions.targetMuscles.map((muscle, index) => (
                    <span 
                      key={index}
                      className="px-3 py-1 bg-gradient-to-r from-[#9234EA]/20 to-[#2565EB]/20 text-[#22D2EE] rounded-full text-xs font-medium border border-[#9234EA]/30"
                    >
                      {muscle}
                    </span>
                  ))}
                </div>
              </div>
            </div>

            {/* Form Instructions */}
            <div className="bg-gradient-to-r from-white/10 to-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
              <h3 className="font-bold text-lg mb-4 text-white flex items-center gap-2">
                <div className="w-2 h-2 bg-[#22D2EE] rounded-full"></div>
                {exercise.instructions.title}
              </h3>
              <ul className="space-y-3 text-sm text-gray-300">
                {exercise.instructions.tips.map((tip, index) => (
                  <li key={index} className="flex items-start gap-3">
                    <span className="bg-gradient-to-r from-[#2565EB] to-[#9234EA] text-white rounded-full w-6 h-6 flex items-center justify-center text-xs font-bold flex-shrink-0">
                      {index + 1}
                    </span>
                    <span className="leading-relaxed">{tip}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* AI Feedback Panel */}
            <div className="bg-gradient-to-r from-white/10 to-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
              <h3 className="font-bold text-lg mb-4 text-white flex items-center gap-2">
                <div className="w-2 h-2 bg-[#34D399] rounded-full animate-pulse"></div>
                AI Feedback
              </h3>
              <div className="space-y-3">
                <div className="bg-[#06976A]/20 border border-[#06976A]/30 rounded-lg p-3">
                  <div className="flex items-center gap-2 mb-1">
                    <div className="w-2 h-2 bg-[#34D399] rounded-full"></div>
                    <span className="text-xs font-medium text-[#34D399]">READY</span>
                  </div>
                  <p className="text-sm text-gray-300">Position yourself in front of the camera to begin</p>
                </div>
                <div className="bg-[#2565EB]/20 border border-[#2565EB]/30 rounded-lg p-3">
                  <div className="flex items-center gap-2 mb-1">
                    <div className="w-2 h-2 bg-[#22D2EE] rounded-full"></div>
                    <span className="text-xs font-medium text-[#22D2EE]">TIP</span>
                  </div>
                  <p className="text-sm text-gray-300">Make sure your full body is visible in the frame</p>
                </div>
                <div className="bg-[#9234EA]/20 border border-[#9234EA]/30 rounded-lg p-3">
                  <div className="flex items-center gap-2 mb-1">
                    <TrendingUp className="w-3 h-3 text-[#9234EA]" />
                    <span className="text-xs font-medium text-[#9234EA]">PROGRESS</span>
                  </div>
                  <p className="text-sm text-gray-300">Start your exercise to receive real-time form analysis</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ExerciseLayout;