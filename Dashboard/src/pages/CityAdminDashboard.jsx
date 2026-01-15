import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useStateContext } from '../contexts/ContextProvider';
import { FaUserShield, FaClipboardCheck, FaClock, FaStar, FaUsers } from 'react-icons/fa';
import { MdAdminPanelSettings } from 'react-icons/md';

const CityAdminDashboard = () => {
  const { currentColor } = useStateContext();
  const navigate = useNavigate();

  // Form state
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    zone: ''
  });

  // Dummy zone admins (pre-added)
  const [zoneAdmins, setZoneAdmins] = useState([
    {
      id: 1,
      name: 'Zone 1 Admin',
      email: 'zone1@nagpur.gov.in',
      zone: 'Dharampeth (Zone 1)',
      totalComplaints: 4,
      resolved: 4,
      avgResolutionTime: '9 days',
      rating: 5
    },
    {
      id: 2,
      name: 'Zone 2 Admin',
      email: 'zone2@nagpur.gov.in',
      zone: 'Laxmi Nagar (Zone 2)',
      totalComplaints: 5,
      resolved: 4,
      avgResolutionTime: '7 days',
      rating: 4.5
    },
    {
      id: 3,
      name: 'Zone 3 Admin',
      email: 'zone3@nagpur.gov.in',
      zone: 'Hanuman Nagar (Zone 3)',
      totalComplaints: 8,
      resolved: 7,
      avgResolutionTime: '6 days',
      rating: 4.8
    },
    {
      id: 4,
      name: 'Zone 4 Admin',
      email: 'zone4@nagpur.gov.in',
      zone: 'Dhantoli (Zone 4)',
      totalComplaints: 3,
      resolved: 3,
      avgResolutionTime: '5 days',
      rating: 5
    },
    {
      id: 5,
      name: 'Zone 5 Admin',
      email: 'zone5@nagpur.gov.in',
      zone: 'Nehru Nagar (Zone 5)',
      totalComplaints: 6,
      resolved: 5,
      avgResolutionTime: '8 days',
      rating: 4.2
    },
    {
      id: 6,
      name: 'Zone 6 Admin',
      email: 'zone6@nagpur.gov.in',
      zone: 'Gandhi Bagh (Zone 6)',
      totalComplaints: 4,
      resolved: 3,
      avgResolutionTime: '10 days',
      rating: 4.0
    }
  ]);

  const handleInputChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (formData.name && formData.email && formData.zone) {
      const newAdmin = {
        id: zoneAdmins.length + 1,
        name: formData.name,
        email: formData.email,
        zone: formData.zone,
        totalComplaints: 0,
        resolved: 0,
        avgResolutionTime: 'N/A',
        rating: 0
      };
      setZoneAdmins([...zoneAdmins, newAdmin]);
      setFormData({ name: '', email: '', zone: '' });
      alert('Zone Admin added successfully! (Note: This is temporary and will not persist on refresh)');
    }
  };

  const handleAdminClick = (admin) => {
    navigate('/zone-admin-performance', { state: { admin } });
  };

  const renderStars = (rating) => {
    return Array.from({ length: 5 }, (_, i) => (
      <FaStar key={i} className={i < rating ? 'text-yellow-400' : 'text-gray-300'} />
    ));
  };

  return (
    <div className="mt-8 px-4 md:px-6">
      {/* Welcome Banner */}
      <div className="mb-8 p-6 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 dark:from-blue-700 dark:to-indigo-700 shadow-lg">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl md:text-3xl font-bold text-white mb-2">
              Welcome, City Admin
            </h2>
            <div className="flex items-center gap-2">
              <span className="px-4 py-1.5 bg-white/20 backdrop-blur-sm rounded-full text-white font-semibold text-sm">
                🏙️ Assigned City: Nagpur
              </span>
              <span className="px-4 py-1.5 bg-white/20 backdrop-blur-sm rounded-full text-white font-semibold text-sm">
                <MdAdminPanelSettings className="inline mr-1" />
                City Administrator
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
          City Admin Dashboard
        </h1>
        <p className="text-gray-600 dark:text-gray-300">
          Monitor zone administrators and their performance across Nagpur
        </p>
      </div>

      {/* Create Zone Admin Form */}
      <div className="bg-white dark:bg-secondary-dark-bg rounded-xl shadow-lg p-6 mb-8">
        <h2 className="text-xl font-bold text-gray-800 dark:text-white mb-4 flex items-center gap-2">
          <FaUserShield style={{ color: currentColor }} />
          Create Zone Admin
        </h2>
        <form onSubmit={handleSubmit} className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
              Name
            </label>
            <input
              type="text"
              name="name"
              value={formData.name}
              onChange={handleInputChange}
              className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-800 dark:text-white"
              placeholder="Enter admin name"
              required
            />
          </div>
          <div>
            <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
              Email
            </label>
            <input
              type="email"
              name="email"
              value={formData.email}
              onChange={handleInputChange}
              className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-800 dark:text-white"
              placeholder="Enter admin email"
              required
            />
          </div>
          <div>
            <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
              Zone
            </label>
            <select
              name="zone"
              value={formData.zone}
              onChange={handleInputChange}
              className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-800 dark:text-white"
              required
            >
              <option value="">Select Zone</option>
              <option value="Dharampeth (Zone 1)">Dharampeth (Zone 1)</option>
              <option value="Laxmi Nagar (Zone 2)">Laxmi Nagar (Zone 2)</option>
              <option value="Hanuman Nagar (Zone 3)">Hanuman Nagar (Zone 3)</option>
              <option value="Dhantoli (Zone 4)">Dhantoli (Zone 4)</option>
              <option value="Nehru Nagar (Zone 5)">Nehru Nagar (Zone 5)</option>
              <option value="Gandhi Bagh (Zone 6)">Gandhi Bagh (Zone 6)</option>
              <option value="Sataranjipura (Zone 7)">Sataranjipura (Zone 7)</option>
              <option value="Lakadganj (Zone 8)">Lakadganj (Zone 8)</option>
              <option value="Ashi Nagar (Zone 9)">Ashi Nagar (Zone 9)</option>
              <option value="Mangalwari (Zone 10)">Mangalwari (Zone 10)</option>
            </select>
          </div>
          <div className="md:col-span-3">
            <button
              type="submit"
              className="px-6 py-2 text-white rounded-lg font-semibold hover:opacity-90 transition-opacity"
              style={{ backgroundColor: currentColor }}
            >
              Submit
            </button>
          </div>
        </form>
      </div>

      {/* Zone Admins Overview */}
      <div className="mb-8">
        <h2 className="text-2xl font-bold text-gray-800 dark:text-white mb-4 flex items-center gap-2">
          <FaUsers style={{ color: currentColor }} />
          Zone Administrators
        </h2>
        
        {/* Statistics Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-white dark:bg-secondary-dark-bg rounded-lg shadow p-4 border-l-4 border-blue-500">
            <p className="text-gray-600 dark:text-gray-400 text-sm">Total Zones</p>
            <p className="text-2xl font-bold text-blue-600">{zoneAdmins.length}</p>
          </div>
          <div className="bg-white dark:bg-secondary-dark-bg rounded-lg shadow p-4 border-l-4 border-green-500">
            <p className="text-gray-600 dark:text-gray-400 text-sm">Total Complaints</p>
            <p className="text-2xl font-bold text-green-600">
              {zoneAdmins.reduce((sum, admin) => sum + admin.totalComplaints, 0)}
            </p>
          </div>
          <div className="bg-white dark:bg-secondary-dark-bg rounded-lg shadow p-4 border-l-4 border-purple-500">
            <p className="text-gray-600 dark:text-gray-400 text-sm">Resolved</p>
            <p className="text-2xl font-bold text-purple-600">
              {zoneAdmins.reduce((sum, admin) => sum + admin.resolved, 0)}
            </p>
          </div>
          <div className="bg-white dark:bg-secondary-dark-bg rounded-lg shadow p-4 border-l-4 border-yellow-500">
            <p className="text-gray-600 dark:text-gray-400 text-sm">Avg Rating</p>
            <p className="text-2xl font-bold text-yellow-600">
              {(zoneAdmins.reduce((sum, admin) => sum + admin.rating, 0) / zoneAdmins.length).toFixed(1)}
            </p>
          </div>
        </div>

        {/* Zone Admins Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {zoneAdmins.map((admin) => (
            <div
              key={admin.id}
              onClick={() => handleAdminClick(admin)}
              className="bg-white dark:bg-secondary-dark-bg rounded-xl shadow-lg p-6 cursor-pointer hover:shadow-xl transform hover:-translate-y-1 transition-all duration-200"
            >
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center gap-3">
                  <div className="p-3 rounded-full" style={{ backgroundColor: `${currentColor}20` }}>
                    <FaUserShield className="text-2xl" style={{ color: currentColor }} />
                  </div>
                  <div>
                    <h3 className="font-bold text-gray-800 dark:text-white text-lg">
                      {admin.name}
                    </h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400">{admin.email}</p>
                  </div>
                </div>
              </div>

              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Zone:</span>
                  <span className="font-semibold text-gray-800 dark:text-white">{admin.zone}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600 dark:text-gray-400 flex items-center gap-1">
                    <FaClipboardCheck />
                    Complaints:
                  </span>
                  <span className="font-semibold text-gray-800 dark:text-white">
                    {admin.resolved}/{admin.totalComplaints}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600 dark:text-gray-400 flex items-center gap-1">
                    <FaClock />
                    Avg Time:
                  </span>
                  <span className="font-semibold text-gray-800 dark:text-white">
                    {admin.avgResolutionTime}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Rating:</span>
                  <div className="flex gap-1">
                    {renderStars(admin.rating)}
                  </div>
                </div>
              </div>

              <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
                <button
                  className="w-full py-2 text-white rounded-lg font-semibold text-sm hover:opacity-90 transition-opacity"
                  style={{ backgroundColor: currentColor }}
                >
                  View Performance →
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Info Note */}
      <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
        <p className="text-sm text-blue-800 dark:text-blue-200">
          <strong>Note:</strong> This is a Proof of Concept (PoC) dashboard using static/dummy data. 
          New zone admins added via the form will only appear temporarily and will not persist on page refresh.
        </p>
      </div>
    </div>
  );
};

export default CityAdminDashboard;
