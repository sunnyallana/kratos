import { ChevronLeft } from 'lucide-react';

type ExerciseLayoutProps = {
  exercise: {
    name: string;
    component: JSX.Element;
    gif: string;
  };
  onBack: () => void;
};

const ExerciseLayout = ({ exercise, onBack }: ExerciseLayoutProps) => {
  return (
    <div className="max-w-7xl mx-auto p-6">
      <button
        onClick={onBack}
        className="flex items-center gap-2 text-blue-600 hover:text-blue-800 mb-6 transition-colors"
      >
        <ChevronLeft className="w-5 h-5" />
        Back to Exercises
      </button>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Video and Camera Feed */}
        <div className="lg:col-span-3 space-y-6">
          {/* Camera and Pose Detection */}
          {exercise.component}
        </div>

        {/* Exercise Instructions */}
        <div className="lg:col-span-1 space-y-6">
          <div className="bg-white rounded-xl p-6 shadow-lg">
            <h2 className="text-2xl font-bold text-gray-800 mb-2">{exercise.name}</h2>
            <div className="aspect-w-16 aspect-h-9 bg-gray-100 rounded-lg overflow-hidden mb-4">
              <img 
                src={exercise.gif} 
                alt={`How to perform ${exercise.name}`}
                className="w-full h-full object-cover"
              />
            </div>
            <h3 className="font-bold text-lg mb-2 text-gray-800">Proper Form Tips</h3>
            <ul className="space-y-2 text-sm text-gray-600">
              <li className="flex items-start gap-2">
                <span className="bg-blue-100 text-blue-600 rounded-full w-5 h-5 flex items-center justify-center text-xs font-bold">1</span>
                Stand straight with shoulders back
              </li>
              <li className="flex items-start gap-2">
                <span className="bg-blue-100 text-blue-600 rounded-full w-5 h-5 flex items-center justify-center text-xs font-bold">2</span>
                Keep your core engaged throughout
              </li>
              <li className="flex items-start gap-2">
                <span className="bg-blue-100 text-blue-600 rounded-full w-5 h-5 flex items-center justify-center text-xs font-bold">3</span>
                Move slowly and with control
              </li>
              <li className="flex items-start gap-2">
                <span className="bg-blue-100 text-blue-600 rounded-full w-5 h-5 flex items-center justify-center text-xs font-bold">4</span>
                Breathe out during exertion
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ExerciseLayout;