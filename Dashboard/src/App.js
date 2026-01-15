import { TooltipComponent } from '@syncfusion/ej2-react-popups';
import React, { useEffect } from 'react';
import { FiSettings } from 'react-icons/fi';
import { BrowserRouter, Route, Routes } from 'react-router-dom';

import './App.css';
import { Footer, Navbar, Sidebar, ThemeSettings, AlertSystem } from './components';
import ErrorBoundary from './components/ErrorBoundary';
import { ColorPicker, ComplainsData, Dashboard, WaterQuality, Editor, SensorAllocation, HealthDataCollection, OutbreakRisk, WaterQualitySensors, ComplaintsAwareness, ComplaintForm, SubmittedComplaints, Login, MyComplaints, AdminDashboard, SubmittedHealthReports, PortableDeviceGuide, CollectedHealthData, UserDashboard, AshaWorkerDashboard, CityAdminDashboard, ZoneAdminPerformance } from './pages';

import { useStateContext } from './contexts/ContextProvider';
import { Navigate } from 'react-router-dom';

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
  
  // Protected Route Component
  const ProtectedRoute = ({ children }) => {
    const userId = localStorage.getItem('userId');
    return userId ? children : <Navigate to="/login" replace />;
  };
  
  // Redirect based on authentication and role
  const LandingRedirect = () => {
    const userId = localStorage.getItem('userId');
    const userRole = localStorage.getItem('userRole');
    
    if (!userId) {
      return <Navigate to="/login" replace />;
    }
    
    if (userRole === 'admin') {
      return <Navigate to="/dashboard" replace />;
    } else if (userRole === 'asha-worker') {
      return <Navigate to="/asha-dashboard" replace />;
    } else if (userRole === 'user') {
      return <Navigate to="/user-dashboard" replace />;
    }
    
    return <Navigate to="/login" replace />;
  };

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
                  {/* Authentication */}
                  <Route path="/login" element={<Login />} />
                  
                  {/* Landing - Redirect based on auth status */}
                  <Route path="/" element={<LandingRedirect />} />
                  
                  {/* dashboard  */}
                  <Route path="/dashboard" element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />
                  <Route path="/user-dashboard" element={<ProtectedRoute><UserDashboard /></ProtectedRoute>} />
                  <Route path="/asha-dashboard" element={<ProtectedRoute><AshaWorkerDashboard /></ProtectedRoute>} />
                  <Route path="/city-admin" element={<ProtectedRoute><CityAdminDashboard /></ProtectedRoute>} />
                  <Route path="/zone-admin-performance" element={<ProtectedRoute><ZoneAdminPerformance /></ProtectedRoute>} />
                  <Route path="/Water-Quality" element={<ProtectedRoute><WaterQuality /></ProtectedRoute>} />

                  {/* pages  */}
                  <Route path="/health-data" element={<ProtectedRoute><HealthDataCollection /></ProtectedRoute>} />
                  <Route path="/submitted-health-reports" element={<ProtectedRoute><SubmittedHealthReports /></ProtectedRoute>} />
                  <Route path="/portable-device-guide" element={<ProtectedRoute><PortableDeviceGuide /></ProtectedRoute>} />
                  <Route path="/collected-health-data" element={<ProtectedRoute><CollectedHealthData /></ProtectedRoute>} />
                  <Route path="/outbreak-risk" element={<ProtectedRoute><OutbreakRisk /></ProtectedRoute>} />
                  <Route path="/water-sensors" element={<ProtectedRoute><WaterQualitySensors /></ProtectedRoute>} />
                  <Route path="/complaints-awareness" element={<ProtectedRoute><ComplaintsAwareness /></ProtectedRoute>} />
                  <Route path="/complains-data" element={<ProtectedRoute><ComplainsData /></ProtectedRoute>} />
                  <Route path="/submit-complaint" element={<ProtectedRoute><ComplaintForm /></ProtectedRoute>} />
                  <Route path="/view-complaints" element={<ProtectedRoute><SubmittedComplaints /></ProtectedRoute>} />
                  <Route path="/my-complaints" element={<ProtectedRoute><MyComplaints /></ProtectedRoute>} />
                  <Route path="/complaint-management" element={<ProtectedRoute><AdminDashboard /></ProtectedRoute>} />

                  {/* apps  */}
                  <Route path="/editor" element={<ProtectedRoute><Editor /></ProtectedRoute>} />
                  <Route path="/sensor-allocation" element={<ProtectedRoute><SensorAllocation /></ProtectedRoute>} />
                  <Route path="/color-picker" element={<ProtectedRoute><ColorPicker /></ProtectedRoute>} />

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
