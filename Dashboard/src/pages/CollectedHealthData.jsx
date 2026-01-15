import React, { useState, useEffect } from 'react';
import { ref, query, orderByChild, onValue } from 'firebase/database';
import { database } from '../firebaseConfig';
import { useStateContext } from '../contexts/ContextProvider';
import { FaClinicMedical, FaMapMarkerAlt, FaUser, FaCalendarAlt, FaVirus } from 'react-icons/fa';

const CollectedHealthData = () => {
  const { currentColor } = useStateContext();
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedReport, setSelectedReport] = useState(null);
  const [filterWorker, setFilterWorker] = useState('all');
  const [workers, setWorkers] = useState([]);

  useEffect(() => {
    const healthReportsRef = ref(database, 'healthReports');
    const q = query(healthReportsRef, orderByChild('submittedAt'));

    const unsubscribe = onValue(q, (snapshot) => {
      if (snapshot.exists()) {
        const reportsArray = [];
        const workerSet = new Set();

        snapshot.forEach((childSnapshot) => {
          const report = {
            id: childSnapshot.key,
            ...childSnapshot.val()
          };
          reportsArray.push(report);
          if (report.userEmail) {
            workerSet.add(report.userEmail);
          }
        });

        // Sort by submission date, newest first
        reportsArray.sort((a, b) => new Date(b.submittedAt) - new Date(a.submittedAt));
        setReports(reportsArray);
        setWorkers(Array.from(workerSet).sort());
      } else {
        setReports([]);
        setWorkers([]);
      }
      setLoading(false);
    });

    return () => unsubscribe();
  }, []);

  const filteredReports = filterWorker === 'all' 
    ? reports 
    : reports.filter(report => report.userEmail === filterWorker);

  const getStatusColor = (disease) => {
    if (!disease) return 'bg-blue-100 dark:bg-blue-900/20 text-blue-800 dark:text-blue-300';
    return 'bg-red-100 dark:bg-red-900/20 text-red-800 dark:text-red-300';
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  };

  const getDiseaseStats = () => {
    const stats = {
      total: filteredReports.length,
      withDisease: filteredReports.filter(r => r.confirmedDisease).length,
      villages: new Set(filteredReports.map(r => r.villageName)).size,
      symptoms: {}
    };

    filteredReports.forEach(report => {
      report.reportedSymptoms.forEach(symptom => {
        stats.symptoms[symptom] = (stats.symptoms[symptom] || 0) + 1;
      });
    });

    return stats;
  };

  const stats = getDiseaseStats();
  const topSymptoms = Object.entries(stats.symptoms)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 5)
    .map(([symptom, count]) => ({ symptom, count }));

  if (loading) {
    return (
      <div className="mt-24 px-4 md:px-6">
        <div className="max-w-6xl mx-auto">
          <div className="text-center py-8">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-current mx-auto mb-4" style={{ color: currentColor }}></div>
            <p className="text-gray-600 dark:text-gray-400">Loading collected health data...</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="mt-24 px-4 md:px-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-800 dark:text-white mb-2">
            <FaClinicMedical className="inline-block mr-3" style={{ color: currentColor }} />
            Collected Health Data from ASHA Workers
          </h1>
          <p className="text-gray-600 dark:text-gray-300">
            View all health reports collected from field workers across villages
          </p>
        </div>

        {/* Statistics Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          <div className="bg-white dark:bg-secondary-dark-bg rounded-xl shadow-lg p-6 border-l-4" style={{ borderColor: currentColor }}>
            <div className="text-3xl font-bold text-gray-800 dark:text-white">{stats.total}</div>
            <p className="text-gray-600 dark:text-gray-400 text-sm mt-2">Total Reports</p>
          </div>
          <div className="bg-white dark:bg-secondary-dark-bg rounded-xl shadow-lg p-6 border-l-4 border-red-500">
            <div className="text-3xl font-bold text-red-600">{stats.withDisease}</div>
            <p className="text-gray-600 dark:text-gray-400 text-sm mt-2">Cases with Disease</p>
          </div>
          <div className="bg-white dark:bg-secondary-dark-bg rounded-xl shadow-lg p-6 border-l-4 border-green-500">
            <div className="text-3xl font-bold text-green-600">{stats.villages}</div>
            <p className="text-gray-600 dark:text-gray-400 text-sm mt-2">Villages Covered</p>
          </div>
          <div className="bg-white dark:bg-secondary-dark-bg rounded-xl shadow-lg p-6 border-l-4 border-blue-500">
            <div className="text-3xl font-bold text-blue-600">{workers.length}</div>
            <p className="text-gray-600 dark:text-gray-400 text-sm mt-2">ASHA Workers</p>
          </div>
        </div>

        {/* Top Symptoms */}
        {topSymptoms.length > 0 && (
          <div className="bg-white dark:bg-secondary-dark-bg rounded-xl shadow-lg p-6 mb-8">
            <h2 className="text-xl font-bold text-gray-800 dark:text-white mb-4 flex items-center gap-2">
              <FaVirus style={{ color: currentColor }} />
              Top Reported Symptoms
            </h2>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              {topSymptoms.map(({ symptom, count }) => (
                <div key={symptom} className="text-center p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
                  <div className="text-2xl font-bold" style={{ color: currentColor }}>{count}</div>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-1 capitalize">
                    {symptom.replace(/_/g, ' ')}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Filter by Worker */}
        {workers.length > 0 && (
          <div className="mb-6">
            <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
              Filter by ASHA Worker
            </label>
            <select
              value={filterWorker}
              onChange={(e) => setFilterWorker(e.target.value)}
              className="w-full md:w-64 px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:text-white"
            >
              <option value="all">All Workers ({workers.length})</option>
              {workers.map((worker) => {
                const count = reports.filter(r => r.userEmail === worker).length;
                return (
                  <option key={worker} value={worker}>
                    {worker} ({count} reports)
                  </option>
                );
              })}
            </select>
          </div>
        )}

        {/* Reports List */}
        {filteredReports.length === 0 ? (
          <div className="bg-white dark:bg-secondary-dark-bg rounded-xl shadow-lg p-8 text-center">
            <FaClinicMedical className="text-5xl text-gray-300 dark:text-gray-600 mx-auto mb-4" />
            <p className="text-gray-600 dark:text-gray-400 text-lg">
              No health reports found.
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-6">
            {filteredReports.map((report) => (
              <div
                key={report.id}
                className="bg-white dark:bg-secondary-dark-bg rounded-xl shadow-lg overflow-hidden hover:shadow-xl transition-shadow cursor-pointer"
                onClick={() => setSelectedReport(selectedReport?.id === report.id ? null : report)}
              >
                {/* Report Header */}
                <div
                  className="px-6 py-4 flex justify-between items-start"
                  style={{
                    borderLeft: `5px solid ${currentColor}`
                  }}
                >
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      <FaMapMarkerAlt style={{ color: currentColor }} className="text-sm" />
                      <h3 className="text-xl font-bold text-gray-800 dark:text-white">
                        {report.villageName}
                      </h3>
                    </div>
                    <div className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                      <p>
                        <span className="font-semibold">Age Group:</span> {report.patientAgeGroup}
                      </p>
                      <p>
                        <span className="font-semibold">Symptoms:</span> {report.reportedSymptoms.length} reported
                      </p>
                      <p>
                        <span className="font-semibold">ASHA Worker:</span> {report.userEmail}
                      </p>
                      {report.confirmedDisease && (
                        <p>
                          <span className="font-semibold">Disease:</span>{' '}
                          <span className={`px-3 py-1 rounded-full text-xs font-semibold ${getStatusColor(report.confirmedDisease)}`}>
                            {report.confirmedDisease.toUpperCase()}
                          </span>
                        </p>
                      )}
                    </div>
                  </div>
                  <div className="text-right text-sm text-gray-500 dark:text-gray-400">
                    <p className="font-semibold">{formatDate(report.submittedDate)}</p>
                    <p className="text-xs mt-1">{new Date(report.submittedAt).toLocaleTimeString()}</p>
                  </div>
                </div>

                {/* Expanded Details */}
                {selectedReport?.id === report.id && (
                  <div className="px-6 py-4 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      {/* Left Column */}
                      <div>
                        <h4 className="font-bold text-gray-800 dark:text-white mb-3 flex items-center gap-2">
                          <FaVirus style={{ color: currentColor }} />
                          Reported Symptoms
                        </h4>
                        <div className="flex flex-wrap gap-2">
                          {report.reportedSymptoms.length > 0 ? (
                            report.reportedSymptoms.map((symptom) => (
                              <span
                                key={symptom}
                                className="px-3 py-1 rounded-full text-xs font-semibold bg-blue-100 dark:bg-blue-900/20 text-blue-800 dark:text-blue-300"
                              >
                                {symptom.replace(/_/g, ' ').toUpperCase()}
                              </span>
                            ))
                          ) : (
                            <span className="text-gray-500 dark:text-gray-400">No symptoms reported</span>
                          )}
                        </div>

                        <h4 className="font-bold text-gray-800 dark:text-white mt-4 mb-3 flex items-center gap-2">
                          <FaUser style={{ color: currentColor }} />
                          Reporter Information
                        </h4>
                        <div className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                          <p><span className="font-semibold">Name:</span> {report.reporterName}</p>
                          <p><span className="font-semibold">Contact:</span> {report.reporterContact}</p>
                          <p><span className="font-semibold">Email:</span> {report.userEmail}</p>
                        </div>
                      </div>

                      {/* Right Column */}
                      <div>
                        <h4 className="font-bold text-gray-800 dark:text-white mb-3 flex items-center gap-2">
                          <FaCalendarAlt style={{ color: currentColor }} />
                          Report Details
                        </h4>
                        <div className="space-y-2 text-sm text-gray-700 dark:text-gray-300 mb-4">
                          <p><span className="font-semibold">Report Date:</span> {formatDate(report.reportDate)}</p>
                          <p><span className="font-semibold">Submitted:</span> {new Date(report.submittedAt).toLocaleString()}</p>
                        </div>

                        {report.additionalNotes && (
                          <>
                            <h4 className="font-bold text-gray-800 dark:text-white mb-3">
                              Additional Notes
                            </h4>
                            <p className="text-sm text-gray-700 dark:text-gray-300 bg-white dark:bg-secondary-dark-bg p-3 rounded-lg border border-gray-200 dark:border-gray-700">
                              {report.additionalNotes}
                            </p>
                          </>
                        )}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default CollectedHealthData;
