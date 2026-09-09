import { UploadView } from './views/UploadView';
import { AnalysisView } from './views/AnalysisView';
import { useAnalysisId } from './useHashRoute';

export function App() {
  const [analysisId, navigate] = useAnalysisId();

  return (
    <div className="app">
      <header className="app-header">
        <button className="wordmark" onClick={() => navigate(null)} type="button">
          ACED
        </button>
        <p className="tagline">Arm-technique analysis for bouldering video</p>
      </header>

      <main>
        {analysisId ? (
          <AnalysisView id={analysisId} onDeleted={() => navigate(null)} />
        ) : (
          <UploadView onStarted={navigate} />
        )}
      </main>
    </div>
  );
}
