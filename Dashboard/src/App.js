import { TooltipComponent } from '@syncfusion/ej2-react-popups';
import React, { useEffect } from 'react';
import { FiSettings } from 'react-icons/fi';
import { BrowserRouter, Route, Routes } from 'react-router-dom';

import './App.css';
import { Footer, Navbar, Sidebar, ThemeSettings, AlertSystem } from './components';
import ErrorBoundary from './components/ErrorBoundary';
import { ColorPicker, ComplainsData, Dashboard, WaterQuality, Editor, SensorAllocation, HealthDataCollection, OutbreakRisk, WaterQualitySensors, ComplaintsAwareness } from './pages';

import { useStateContext } from './contexts/ContextProvider';

const App = () => {
  const { setCurrentColor, setCurrentMode, currentMode, activeMenu, currentColor, themeSettings, setThemeSettings } = useStateContext();

  useEffect(() => {
    const currentThemeColor = localStorage.getItem('colorMode');
    const currentThemeMode = localStorage.getItem('themeMode');
    if (currentThemeColor && currentThemeMode) {
      setCurrentColor(currentThemeColor);
      setCurrentMode(currentThemeMode);
    }
  }, []);

  return (
    <ErrorBoundary>
      <div className={currentMode === 'Dark' ? 'dark' : ''}>
        <BrowserRouter>
          <div className="flex relative dark:bg-main-dark-bg">
            <div className="fixed right-4 bottom-4" style={{ zIndex: '1000' }}>
              <TooltipComponent
                content="Settings"
                position="Top"
              >
                <button
                  type="button"
                  aria-label="Open settings"
                  onClick={() => setThemeSettings(true)}
                  style={{ background: currentColor, borderRadius: '50%' }}
                  className="text-3xl text-white p-3 hover:drop-shadow-xl hover:bg-light-gray"
                >
                  <FiSettings />
                </button>

              </TooltipComponent>
            </div>
            {activeMenu ? (
              <div className="w-72 fixed sidebar dark:bg-secondary-dark-bg bg-white ">
                <Sidebar />
              </div>
            ) : (
              <div className="w-0 dark:bg-secondary-dark-bg">
                <Sidebar />
              </div>
            )}
            <div
              className={
              activeMenu
                ? 'dark:bg-main-dark-bg  bg-main-bg min-h-screen md:ml-72 w-full  '
                : 'bg-main-bg dark:bg-main-dark-bg  w-full min-h-screen flex-2 '
            }
            >
              <div className="fixed md:static bg-main-bg dark:bg-main-dark-bg navbar w-full ">
                <Navbar />
              </div>
              <div>
                {themeSettings && (<ThemeSettings />)}
                <AlertSystem />

                <Routes>
                  {/* dashboard  */}
                  <Route path="/" element={(<Dashboard />)} />
                  <Route path="/dashboard" element={(<Dashboard />)} />
                  <Route path="/Water-Quality" element={(<WaterQuality />)} />

                  {/* pages  */}
                  <Route path="/health-data" element={<HealthDataCollection />} />
                  <Route path="/outbreak-risk" element={<OutbreakRisk />} />
                  <Route path="/water-sensors" element={<WaterQualitySensors />} />
                  <Route path="/complaints-awareness" element={<ComplaintsAwareness />} />
                  <Route path="/complains-data" element={<ComplainsData />} />

                  {/* apps  */}
                  <Route path="/editor" element={<Editor />} />
                  <Route path="/sensor-allocation" element={<SensorAllocation />} />
                  <Route path="/color-picker" element={<ColorPicker />} />

                  {/* charts  */}
                  {/* <Route path="/line" element={<Line />} />
                <Route path="/area" element={<Area />} />
                <Route path="/bar" element={<Bar />} />
                <Route path="/pie" element={<Pie />} />
                <Route path="/financial" element={<Financial />} />
                <Route path="/color-mapping" element={<ColorMapping />} />
                <Route path="/pyramid" element={<Pyramid />} />
                <Route path="/stacked" element={<Stacked />} /> */}

                </Routes>
              </div>
              <Footer />
            </div>
          </div>
        </BrowserRouter>
      </div>
    </ErrorBoundary>
  );
};

export default App;
