import React, { useState } from 'react';
import { ToastProvider } from './hooks/useToast';
import Navbar from './components/Navbar';
import Login from './pages/Login';
import Dashboard from './pages/Dashboard';
import Detect from './pages/Detect';
import AttackGraph from './pages/AttackGraph';
import Playbook from './pages/Playbook';
import Outcome from './pages/Outcome';
import AISEngine from './pages/AISEngine';
import Alerts from './pages/Alerts';
import Reports from './pages/Reports';

// ─── Section keys ─────────────────────────────────────────────────────────────
// login | dashboard | detect | attack-graph | playbook | outcome | ais | alerts | reports
// Add new pages here and in the switch below

function App() {
  const [currentSection, setCurrentSection] = useState('login');
  const [lastPipelineResult, setLastPipelineResult] = React.useState(null);

  const renderSection = () => {
    switch (currentSection) {
      case 'dashboard':    return <Dashboard navigateTo={setCurrentSection} />;
      case 'detect':       return <Detect onPipelineResult={setLastPipelineResult} />;
      case 'attack-graph': return <AttackGraph />;
      case 'playbook':     return <Playbook lastPipelineResult={lastPipelineResult} />;
      case 'outcome':      return <Outcome />;
      case 'ais':          return <AISEngine />;
      case 'alerts':       return <Alerts />;
      case 'reports':      return <Reports />;
      default:             return null;
    }
  };

  if (currentSection === 'login') {
    return (
      <ToastProvider>
        <Login onLogin={() => setCurrentSection('dashboard')} />
      </ToastProvider>
    );
  }

  return (
    <ToastProvider>
      <div id="app" style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
        <Navbar currentSection={currentSection} setCurrentSection={setCurrentSection} />
        <main style={{ marginLeft: 0, paddingBottom: 60 }}>
          {renderSection()}
        </main>
      </div>
    </ToastProvider>
  );
}

export default App;
