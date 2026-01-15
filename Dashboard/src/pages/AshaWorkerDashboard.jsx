import React, { useEffect, useState } from 'react';
import { ref, query, orderByChild, equalTo, onValue } from 'firebase/database';
import { database } from '../firebaseConfig';
import { useStateContext } from '../contexts/ContextProvider';
import { FaClinicMedical, FaHeartbeat, FaVirus, FaMapMarkedAlt, FaUsers } from 'react-icons/fa';

const AshaWorkerDashboard = () => {
  const { currentColor } = useStateContext();
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(true);

  const userName = localStorage.getItem('userName') || 'ASHA Worker';
  const userZone = localStorage.getItem('userZone') || 'Dharampeth (Zone 2)';
  const userId = localStorage.getItem('userId');

  useEffect(() => {
    if (!userId) {
      setLoading(false);
      return;
    }

    const healthReportsRef = ref(database, 'healthReports');
    const q = query(healthReportsRef, orderByChild('userId'), equalTo(userId));

    const unsubscribe = onValue(q, (snapshot) => {
      if (snapshot.exists()) {
        const reportsArray = [];
        snapshot.forEach((childSnapshot) => {
          reportsArray.push({
            id: childSnapshot.key,
            ...childSnapshot.val()
          });
        });
        setReports(reportsArray);
      } else {
        setReports([]);
      }
      setLoading(false);
    });

    return () => unsubscribe();
  }, [userId]);

  const getHealthStats = () => {
    const villages = new Set(reports.map(r => r.villageName)).size;
    const withDisease = reports.filter(r => r.confirmedDisease).length;
    const symptomsCount = {};
    
    reports.forEach(report => {
      report.reportedSymptoms.forEach(symptom => {
        symptomsCount[symptom] = (symptomsCount[symptom] || 0) + 1;
      });
    });

    const topSymptom = Object.entries(symptomsCount)
      .sort(([, a], [, b]) => b - a)[0];

    return {
      total: reports.length,
      villages,
      withDisease,
      topSymptom: topSymptom ? topSymptom[0].replace(/_/g, ' ') : 'N/A'
    };
  };

  const stats = getHealthStats();

  const recentReports = reports
    .sort((a, b) => new Date(b.submittedAt) - new Date(a.submittedAt))
    .slice(0, 3);

  return (
    <div className="mt-8 px-4 md:px-6">
      {/* Welcome Banner */}
      <div className="mb-8 p-6 rounded-xl bg-gradient-to-r from-pink-500 to-rose-500 dark:from-pink-600 dark:to-rose-600 shadow-lg">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl md:text-3xl font-bold text-white mb-2">
              Welcome, ASHA Worker!
            </h2>
            <div className="flex items-center gap-2">
              <span className="px-4 py-1.5 bg-white/20 backdrop-blur-sm rounded-full text-white font-semibold text-sm">
                📍 {userZone}
              </span>
              <span className="px-4 py-1.5 bg-white/20 backdrop-blur-sm rounded-full text-white font-semibold text-sm">
                🏥 ASHA Worker
              </span>
            </div>
          </div>
          <div className="hidden md:block">
            <div className="text-white text-right">
              <p className="text-sm opacity-90">Current Date</p>
              <p className="text-lg font-semibold">{new Date().toLocaleDateString('en-US', { 
                weekday: 'short', 
                year: 'numeric', 
                month: 'short', 
                day: 'numeric' 
              })}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Main Title */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-slate-800 dark:text-white mb-2">
          ASHA Worker Dashboard
        </h1>
        <p className="text-gray-600 dark:text-gray-300">
          Track your field work and health data collection activities
        </p>
      </div>

      {/* Statistics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <div className="bg-white dark:bg-secondary-dark-bg rounded-xl shadow-lg p-6 border-l-4" style={{ borderColor: currentColor }}>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-600 dark:text-gray-400 text-sm mb-1">Total Reports</p>
              <p className="text-3xl font-bold text-gray-800 dark:text-white">{stats.total}</p>
            </div>
            <div className="p-4 rounded-full" style={{ backgroundColor: `${currentColor}20` }}>
              <FaClinicMedical className="text-2xl" style={{ color: currentColor }} />
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-secondary-dark-bg rounded-xl shadow-lg p-6 border-l-4 border-green-500">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-600 dark:text-gray-400 text-sm mb-1">Villages Covered</p>
              <p className="text-3xl font-bold text-green-600">{stats.villages}</p>
            </div>
            <div className="p-4 bg-green-100 dark:bg-green-900/20 rounded-full">
              <FaMapMarkedAlt className="text-2xl text-green-600" />
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-secondary-dark-bg rounded-xl shadow-lg p-6 border-l-4 border-red-500">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-600 dark:text-gray-400 text-sm mb-1">Disease Cases</p>
              <p className="text-3xl font-bold text-red-600">{stats.withDisease}</p>
            </div>
            <div className="p-4 bg-red-100 dark:bg-red-900/20 rounded-full">
              <FaVirus className="text-2xl text-red-600" />
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-secondary-dark-bg rounded-xl shadow-lg p-6 border-l-4 border-purple-500">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-600 dark:text-gray-400 text-sm mb-1">Top Symptom</p>
              <p className="text-lg font-bold text-purple-600 capitalize">{stats.topSymptom}</p>
            </div>
            <div className="p-4 bg-purple-100 dark:bg-purple-900/20 rounded-full">
              <FaHeartbeat className="text-2xl text-purple-600" />
            </div>
          </div>
        </div>
      </div>

      {/* Recent Health Reports */}
      <div className="bg-white dark:bg-secondary-dark-bg rounded-xl shadow-lg p-6 mb-8">
        <h2 className="text-xl font-bold text-gray-800 dark:text-white mb-4 flex items-center gap-2">
          <FaClinicMedical style={{ color: currentColor }} />
          Recent Health Reports
        </h2>
        
        {loading ? (
          <div className="text-center py-8">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-current mx-auto" style={{ color: currentColor }}></div>
          </div>
        ) : recentReports.length > 0 ? (
          <div className="space-y-4">
            {recentReports.map((report) => (
              <div key={report.id} className="border border-gray-200 dark:border-gray-700 rounded-lg p-4 hover:shadow-md transition-shadow">
                <div className="flex justify-between items-start mb-2">
                  <div className="flex-1">
                    <p className="font-semibold text-gray-800 dark:text-white">📍 {report.villageName}</p>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                      Age Group: {report.patientAgeGroup} | Symptoms: {report.reportedSymptoms.length}
                    </p>
                  </div>
                  {report.confirmedDisease && (
                    <span className="px-3 py-1 bg-red-100 dark:bg-red-900/20 text-red-700 dark:text-red-300 rounded-full text-xs font-semibold">
                      {report.confirmedDisease.toUpperCase()}
                    </span>
                  )}
                </div>
                <div className="flex justify-between items-center text-xs text-gray-500 dark:text-gray-400 mt-3">
                  <span>Reporter: {report.reporterName}</span>
                  <span>{new Date(report.submittedAt).toLocaleDateString()}</span>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8">
            <FaUsers className="text-5xl text-gray-300 dark:text-gray-600 mx-auto mb-3" />
            <p className="text-gray-600 dark:text-gray-400">No health reports yet. Start collecting data in your field visits!</p>
          </div>
        )}
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-gradient-to-br from-teal-500 to-teal-600 rounded-xl shadow-lg p-6 text-white">
          <h3 className="text-xl font-bold mb-2">Collect Health Data</h3>
          <p className="text-teal-100 mb-4">Submit new health reports from field visits</p>
          <button 
            onClick={() => window.location.href = '/health-data'}
            className="bg-white text-teal-600 px-6 py-2 rounded-lg font-semibold hover:bg-teal-50 transition-colors"
          >
            Submit Report
          </button>
        </div>

        <div className="bg-gradient-to-br from-indigo-500 to-indigo-600 rounded-xl shadow-lg p-6 text-white">
          <h3 className="text-xl font-bold mb-2">View Submitted Reports</h3>
          <p className="text-indigo-100 mb-4">Check all your submitted health data</p>
          <button 
            onClick={() => window.location.href = '/submitted-health-reports'}
            className="bg-white text-indigo-600 px-6 py-2 rounded-lg font-semibold hover:bg-indigo-50 transition-colors"
          >
            View Reports
          </button>
        </div>

        <div className="bg-gradient-to-br from-orange-500 to-orange-600 rounded-xl shadow-lg p-6 text-white">
          <h3 className="text-xl font-bold mb-2">Device Guide</h3>
          <p className="text-orange-100 mb-4">Learn to use your portable testing device</p>
          <button 
            onClick={() => window.location.href = '/portable-device-guide'}
            className="bg-white text-orange-600 px-6 py-2 rounded-lg font-semibold hover:bg-orange-50 transition-colors"
          >
            View Guide
          </button>
        </div>
      </div>
    </div>
  );
};

export default AshaWorkerDashboard;
