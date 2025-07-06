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
        return 'text-[#10B981] bg-[#10B981]/20 border-[#10B981]/40';
      case 'intermediate':
        return 'text-[#06B6D4] bg-[#06B6D4]/20 border-[#06B6D4]/40';
      case 'advanced':
        return 'text-[#A855F7] bg-[#A855F7]/20 border-[#A855F7]/40';
      default:
        return 'text-[#3B82F6] bg-[#3B82F6]/20 border-[#3B82F6]/40';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-pink-50 via-purple-50 to-blue-50">
      <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPGcgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KPGcgZmlsbD0iIzhlYTNiYSIgZmlsbC1vcGFjaXR5PSIwLjA1Ij4KPGNpcmNsZSBjeD0iMjkiIGN5PSIyOSIgcj0iMS41Ii8+CjxwYXRoIGQ9Im0yOS01LTUtNWgxMFptMCAxMGwtNS01aDE0bC00IDR6bTAgMTBsLTUtNWgxNGwtNCA0em0wIDEwbC01LTVoMTRsLTQgNHptMCAxMGwtNS01aDE0bC00IDR6bTIwIDEwbC01LTVoMTRsLTQgNHoiLz4KPC9nPgo8L2c+Cjwvc3ZnPg==')] opacity-30"></div>
      
      <div className="relative max-w-7xl mx-auto p-6">
        <button
          onClick={onBack}
          className="flex items-center gap-3 text-gray-700 hover:text-[#06B6D4] mb-8 transition-colors duration-300 group"
        >
          <div className="p-2 bg-white/70 backdrop-blur-lg rounded-full border border-gray-200 group-hover:border-purple-300 transition-all duration-300">
            <ChevronLeft className="w-5 h-5 group-hover:-translate-x-1 transition-transform" />
          </div>
          <span className="font-semibold">Back to Exercises</span>
        </button>

        <div className="grid grid-cols-1 xl:grid-cols-4 gap-8">
          {/* Main Training Area */}
          <div className="xl:col-span-3 space-y-6">
            {/* Training Header */}
            <div className="bg-gradient-to-r from-white/70 to-white/50 backdrop-blur-lg rounded-2xl p-6 border border-white/40 shadow-lg">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <div className={`p-3 rounded-xl bg-gradient-to-br ${exercise.color} text-white shadow-lg`}>
                    <Activity className="w-6 h-6" />
                  </div>
                  <div>
                    <h1 className="text-2xl font-bold text-gray-800">{exercise.name} Training</h1>
                    <p className="text-gray-600">AI-Powered Form Analysis</p>
                  </div>
                </div>
                <div className="flex items-center gap-4">
                  <div className="flex items-center gap-2 text-[#10B981]">
                    <div className="w-2 h-2 bg-[#10B981] rounded-full animate-pulse"></div>
                    <span className="text-sm font-medium">AI Active</span>
                  </div>
                  <div className="flex items-center gap-2 text-[#06B6D4]">
                    <Camera className="w-4 h-4" />
                    <span className="text-sm font-medium">Camera Ready</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Camera and Pose Detection */}
            <div className="bg-gradient-to-r from-white/70 to-white/50 backdrop-blur-lg rounded-2xl p-6 border border-white/40 shadow-lg">
              {exercise.component}
            </div>

            {/* YouTube Video Tutorial - Now Below Camera */}
            <div className="bg-gradient-to-r from-white/70 to-white/50 backdrop-blur-lg rounded-2xl p-6 border border-white/40 shadow-lg">
              <h3 className="font-bold text-xl mb-4 text-gray-800 flex items-center gap-3">
                <div className="p-2 bg-gradient-to-r from-[#3B82F6] to-[#A855F7] rounded-lg">
                  <Play className="w-5 h-5 text-white" />
                </div>
                Video Tutorial - {exercise.name}
              </h3>
              <div className="relative aspect-video bg-gray-100 rounded-xl overflow-hidden border border-gray-200 shadow-lg">
                <iframe
                  src={`https://www.youtube.com/embed/${exercise.youtubeId}?rel=0&modestbranding=1&showinfo=0`}
                  title={`${exercise.name} Tutorial`}
                  className="w-full h-full"
                  frameBorder="0"
                  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                  allowFullScreen
                />
              </div>
              <div className="mt-4 p-4 bg-gradient-to-r from-[#3B82F6]/10 to-[#A855F7]/10 rounded-lg border border-[#3B82F6]/20">
                <p className="text-gray-700 leading-relaxed">
                  <span className="text-[#06B6D4] font-semibold">Watch first:</span> This professional demonstration shows proper {exercise.name.toLowerCase()} technique. Study the form before starting your AI-guided training session for best results.
                </p>
              </div>
            </div>
          </div>

          {/* Exercise Instructions Sidebar */}
          <div className="xl:col-span-1 space-y-6">
            {/* Exercise Info Card */}
            <div className="bg-gradient-to-r from-white/70 to-white/50 backdrop-blur-lg rounded-2xl p-6 border border-white/40 shadow-lg">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">{exercise.name}</h2>
              <div className="aspect-w-16 aspect-h-9 bg-gradient-to-br from-white/60 to-gray-100/60 rounded-xl overflow-hidden mb-6 border border-gray-200">
                <img 
                  src={exercise.gif} 
                  alt={`How to perform ${exercise.name}`}
                  className="w-full h-full object-cover"
                />
              </div>

              {/* Target Muscles */}
              <div className="mb-6">
                <h4 className="text-sm font-semibold text-gray-800 mb-3 flex items-center gap-2">
                  <Zap className="w-4 h-4 text-[#8B5CF6]" />
                  Target Muscles
                </h4>
                <div className="flex flex-wrap gap-2">
                  {exercise.instructions.targetMuscles.map((muscle, index) => (
                    <span 
                      key={index}
                      className="px-3 py-1 bg-gradient-to-r from-[#A855F7]/20 to-[#3B82F6]/20 text-[#06B6D4] rounded-full text-xs font-medium border border-[#A855F7]/30"
                    >
                      {muscle}
                    </span>
                  ))}
                </div>
              </div>
            </div>

            {/* Form Instructions */}
            <div className="bg-gradient-to-r from-white/70 to-white/50 backdrop-blur-lg rounded-2xl p-6 border border-white/40 shadow-lg">
              <h3 className="font-bold text-lg mb-4 text-gray-800 flex items-center gap-2">
                <div className="w-2 h-2 bg-[#06B6D4] rounded-full"></div>
                {exercise.instructions.title}
              </h3>
              <ul className="space-y-3 text-sm text-gray-700">
                {exercise.instructions.tips.map((tip, index) => (
                  <li key={index} className="flex items-start gap-3">
                    <span className="bg-gradient-to-r from-[#3B82F6] to-[#A855F7] text-white rounded-full w-6 h-6 flex items-center justify-center text-xs font-bold flex-shrink-0">
                      {index + 1}
                    </span>
                    <span className="leading-relaxed">{tip}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* AI Feedback Panel */}
            <div className="bg-gradient-to-r from-white/70 to-white/50 backdrop-blur-lg rounded-2xl p-6 border border-white/40 shadow-lg">
              <h3 className="font-bold text-lg mb-4 text-gray-800 flex items-center gap-2">
                <div className="w-2 h-2 bg-[#10B981] rounded-full animate-pulse"></div>
                AI Feedback
              </h3>
              <div className="space-y-3">
                <div className="bg-[#10B981]/20 border border-[#10B981]/40 rounded-lg p-3">
                  <div className="flex items-center gap-2 mb-1">
                    <div className="w-2 h-2 bg-[#10B981] rounded-full"></div>
                    <span className="text-xs font-medium text-[#10B981]">READY</span>
                  </div>
                  <p className="text-sm text-gray-700">Position yourself in front of the camera to begin</p>
                </div>
                <div className="bg-[#3B82F6]/20 border border-[#3B82F6]/40 rounded-lg p-3">
                  <div className="flex items-center gap-2 mb-1">
                    <div className="w-2 h-2 bg-[#06B6D4] rounded-full"></div>
                    <span className="text-xs font-medium text-[#06B6D4]">TIP</span>
                  </div>
                  <p className="text-sm text-gray-700">Make sure your full body is visible in the frame</p>
                </div>
                <div className="bg-[#A855F7]/20 border border-[#A855F7]/40 rounded-lg p-3">
                  <div className="flex items-center gap-2 mb-1">
                    <TrendingUp className="w-3 h-3 text-[#A855F7]" />
                    <span className="text-xs font-medium text-[#A855F7]">PROGRESS</span>
                  </div>
                  <p className="text-sm text-gray-700">Start your exercise to receive real-time form analysis</p>
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