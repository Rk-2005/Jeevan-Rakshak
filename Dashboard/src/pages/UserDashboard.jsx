import React, { useEffect, useState } from 'react';
import { ref, query, orderByChild, equalTo, onValue } from 'firebase/database';
import { database } from '../firebaseConfig';
import { useStateContext } from '../contexts/ContextProvider';
import { FaClipboardList, FaCheckCircle, FaClock, FaExclamationTriangle, FaTint } from 'react-icons/fa';

const UserDashboard = () => {
  const { currentColor } = useStateContext();
  const [complaints, setComplaints] = useState([]);
  const [loading, setLoading] = useState(true);

  const userName = localStorage.getItem('userName') || 'User';
  const userZone = localStorage.getItem('userZone') || 'Dharampeth (Zone 2)';
  const userId = localStorage.getItem('userId');

  useEffect(() => {
    if (!userId) {
      setLoading(false);
      return;
    }

    const complaintsRef = ref(database, 'complaints');
    const q = query(complaintsRef, orderByChild('userId'), equalTo(userId));

    const unsubscribe = onValue(q, (snapshot) => {
      if (snapshot.exists()) {
        const complaintsArray = [];
        snapshot.forEach((childSnapshot) => {
          complaintsArray.push({
            id: childSnapshot.key,
            ...childSnapshot.val()
          });
        });
        setComplaints(complaintsArray);
      } else {
        setComplaints([]);
      }
      setLoading(false);
    });

    return () => unsubscribe();
  }, [userId]);

  const getStatusCounts = () => {
    return {
      total: complaints.length,
      pending: complaints.filter(c => c.status === 'pending').length,
      inProgress: complaints.filter(c => c.status === 'in-progress' || c.status === 'reviewed').length,
      resolved: complaints.filter(c => c.status === 'closed' || c.status === 'solved').length,
    };
  };

  const stats = getStatusCounts();

  const getPriorityColor = (priority) => {
    switch (priority?.toLowerCase()) {
      case 'high': return 'text-red-600 bg-red-100 dark:bg-red-900/20';
      case 'medium': return 'text-yellow-600 bg-yellow-100 dark:bg-yellow-900/20';
      case 'low': return 'text-green-600 bg-green-100 dark:bg-green-900/20';
      default: return 'text-gray-600 bg-gray-100 dark:bg-gray-900/20';
    }
  };

  const recentComplaints = complaints
    .sort((a, b) => new Date(b.submittedAt) - new Date(a.submittedAt))
    .slice(0, 3);

  return (
    <div className="mt-8 px-4 md:px-6">
      {/* Welcome Banner */}
      <div className="mb-8 p-6 rounded-xl bg-gradient-to-r from-green-500 to-teal-500 dark:from-green-600 dark:to-teal-600 shadow-lg">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl md:text-3xl font-bold text-white mb-2">
              Welcome, User!
            </h2>
            <div className="flex items-center gap-2">
              <span className="px-4 py-1.5 bg-white/20 backdrop-blur-sm rounded-full text-white font-semibold text-sm">
                📍 {userZone}
              </span>
              <span className="px-4 py-1.5 bg-white/20 backdrop-blur-sm rounded-full text-white font-semibold text-sm">
                👤 User
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
          My Dashboard
        </h1>
        <p className="text-gray-600 dark:text-gray-300">
          Track your complaints and stay updated on community health
        </p>
      </div>

      {/* Statistics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <div className="bg-white dark:bg-secondary-dark-bg rounded-xl shadow-lg p-6 border-l-4" style={{ borderColor: currentColor }}>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-600 dark:text-gray-400 text-sm mb-1">Total Complaints</p>
              <p className="text-3xl font-bold text-gray-800 dark:text-white">{stats.total}</p>
            </div>
            <div className="p-4 rounded-full" style={{ backgroundColor: `${currentColor}20` }}>
              <FaClipboardList className="text-2xl" style={{ color: currentColor }} />
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-secondary-dark-bg rounded-xl shadow-lg p-6 border-l-4 border-yellow-500">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-600 dark:text-gray-400 text-sm mb-1">Pending</p>
              <p className="text-3xl font-bold text-yellow-600">{stats.pending}</p>
            </div>
            <div className="p-4 bg-yellow-100 dark:bg-yellow-900/20 rounded-full">
              <FaClock className="text-2xl text-yellow-600" />
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-secondary-dark-bg rounded-xl shadow-lg p-6 border-l-4 border-blue-500">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-600 dark:text-gray-400 text-sm mb-1">In Progress</p>
              <p className="text-3xl font-bold text-blue-600">{stats.inProgress}</p>
            </div>
            <div className="p-4 bg-blue-100 dark:bg-blue-900/20 rounded-full">
              <FaExclamationTriangle className="text-2xl text-blue-600" />
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-secondary-dark-bg rounded-xl shadow-lg p-6 border-l-4 border-green-500">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-600 dark:text-gray-400 text-sm mb-1">Resolved</p>
              <p className="text-3xl font-bold text-green-600">{stats.resolved}</p>
            </div>
            <div className="p-4 bg-green-100 dark:bg-green-900/20 rounded-full">
              <FaCheckCircle className="text-2xl text-green-600" />
            </div>
          </div>
        </div>
      </div>

      {/* Recent Complaints */}
      <div className="bg-white dark:bg-secondary-dark-bg rounded-xl shadow-lg p-6 mb-8">
        <h2 className="text-xl font-bold text-gray-800 dark:text-white mb-4 flex items-center gap-2">
          <FaClipboardList style={{ color: currentColor }} />
          Recent Complaints
        </h2>
        
        {loading ? (
          <div className="text-center py-8">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-current mx-auto" style={{ color: currentColor }}></div>
          </div>
        ) : recentComplaints.length > 0 ? (
          <div className="space-y-4">
            {recentComplaints.map((complaint) => (
              <div key={complaint.id} className="border border-gray-200 dark:border-gray-700 rounded-lg p-4 hover:shadow-md transition-shadow">
                <div className="flex justify-between items-start mb-2">
                  <div className="flex-1">
                    <p className="font-semibold text-gray-800 dark:text-white capitalize">{complaint.type}</p>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">{complaint.description}</p>
                  </div>
                  <span className={`px-3 py-1 rounded-full text-xs font-semibold ${getPriorityColor(complaint.priority)}`}>
                    {complaint.priority}
                  </span>
                </div>
                <div className="flex justify-between items-center text-xs text-gray-500 dark:text-gray-400 mt-3">
                  <span>📍 {complaint.zone}</span>
                  <span className={`px-2 py-1 rounded ${
                    complaint.status === 'closed' || complaint.status === 'solved' 
                      ? 'bg-green-100 text-green-700 dark:bg-green-900/20 dark:text-green-300'
                      : complaint.status === 'pending'
                      ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/20 dark:text-yellow-300'
                      : 'bg-blue-100 text-blue-700 dark:bg-blue-900/20 dark:text-blue-300'
                  }`}>
                    {complaint.status}
                  </span>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8">
            <FaTint className="text-5xl text-gray-300 dark:text-gray-600 mx-auto mb-3" />
            <p className="text-gray-600 dark:text-gray-400">No complaints yet. Submit your first complaint to get started!</p>
          </div>
        )}
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl shadow-lg p-6 text-white">
          <h3 className="text-xl font-bold mb-2">Submit a Complaint</h3>
          <p className="text-blue-100 mb-4">Report water quality issues in your community</p>
          <button 
            onClick={() => window.location.href = '/submit-complaint'}
            className="bg-white text-blue-600 px-6 py-2 rounded-lg font-semibold hover:bg-blue-50 transition-colors"
          >
            Submit Now
          </button>
        </div>

        <div className="bg-gradient-to-br from-purple-500 to-purple-600 rounded-xl shadow-lg p-6 text-white">
          <h3 className="text-xl font-bold mb-2">View All Complaints</h3>
          <p className="text-purple-100 mb-4">Track status of all your submitted complaints</p>
          <button 
            onClick={() => window.location.href = '/my-complaints'}
            className="bg-white text-purple-600 px-6 py-2 rounded-lg font-semibold hover:bg-purple-50 transition-colors"
          >
            View All
          </button>
        </div>
      </div>
    </div>
  );
};

export default UserDashboard;
