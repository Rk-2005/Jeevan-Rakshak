import React, { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { useStateContext } from '../contexts/ContextProvider';
import { 
  FaUserShield, 
  FaClipboardCheck, 
  FaClock, 
  FaStar, 
  FaCheckCircle, 
  FaExclamationCircle,
  FaPhone,
  FaMapMarkerAlt,
  FaArrowLeft,
  FaKey,
  FaCopy,
  FaEnvelope,
  FaLock,
  FaEye,
  FaEyeSlash
} from 'react-icons/fa';
import { MdTimeline } from 'react-icons/md';

const ZoneAdminPerformance = () => {
  const { currentColor } = useStateContext();
  const location = useLocation();
  const navigate = useNavigate();
  const admin = location.state?.admin;

  // If no admin data, redirect back
  if (!admin) {
    navigate('/city-admin');
    return null;
  }

  // Dummy complaints data (4 complaints for Zone 1 Admin)
  const complaintsData = admin.zone === 'Dharampeth (Zone 1)' ? [
    {
      id: 1,
      submittedBy: 'Jayesh K',
      phone: '8988229922',
      zone: 'Dharampeth (Zone 1)',
      address: 'Near Zone 1 Opposite of Jr Road',
      description: 'heavy leakage',
      timeline: [
        { date: '11 Dec 2025', status: 'Complaint Raised', icon: FaExclamationCircle, color: 'red' },
        { date: '15 Dec 2025', status: 'Taken by Admin', icon: FaClock, color: 'yellow' },
        { date: '20 Dec 2025', status: 'Closed', icon: FaCheckCircle, color: 'green' }
      ],
      userFeedback: {
        rating: 5,
        comment: 'Very good service'
      }
    },
    {
      id: 2,
      submittedBy: 'Priya Sharma',
      phone: '9876543210',
      zone: 'Dharampeth (Zone 1)',
      address: 'Main Road, Dharampeth Extension',
      description: 'Water contamination - bad smell and color',
      timeline: [
        { date: '05 Dec 2025', status: 'Complaint Raised', icon: FaExclamationCircle, color: 'red' },
        { date: '07 Dec 2025', status: 'Taken by Admin', icon: FaClock, color: 'yellow' },
        { date: '12 Dec 2025', status: 'Closed', icon: FaCheckCircle, color: 'green' }
      ],
      userFeedback: {
        rating: 5,
        comment: 'Excellent response time'
      }
    },
    {
      id: 3,
      submittedBy: 'Rahul Deshmukh',
      phone: '9123456789',
      zone: 'Dharampeth (Zone 1)',
      address: 'Gandhi Nagar, Behind Temple',
      description: 'Broken water pipeline causing road flooding',
      timeline: [
        { date: '18 Dec 2025', status: 'Complaint Raised', icon: FaExclamationCircle, color: 'red' },
        { date: '20 Dec 2025', status: 'Taken by Admin', icon: FaClock, color: 'yellow' },
        { date: '28 Dec 2025', status: 'Closed', icon: FaCheckCircle, color: 'green' }
      ],
      userFeedback: {
        rating: 4,
        comment: 'Good work, but took a bit longer'
      }
    },
    {
      id: 4,
      submittedBy: 'Sunita Patel',
      phone: '9988776655',
      zone: 'Dharampeth (Zone 1)',
      address: 'Residential Complex, Sector A',
      description: 'Low water pressure in entire building',
      timeline: [
        { date: '22 Dec 2025', status: 'Complaint Raised', icon: FaExclamationCircle, color: 'red' },
        { date: '24 Dec 2025', status: 'Taken by Admin', icon: FaClock, color: 'yellow' },
        { date: '30 Dec 2025', status: 'Closed', icon: FaCheckCircle, color: 'green' }
      ],
      userFeedback: {
        rating: 5,
        comment: 'Very satisfied with the solution'
      }
    }
  ] : [
    {
      id: 1,
      submittedBy: 'Sample User',
      phone: '9876543210',
      zone: admin.zone,
      address: 'Sample Address',
      description: 'Sample complaint description',
      timeline: [
        { date: '10 Jan 2026', status: 'Complaint Raised', icon: FaExclamationCircle, color: 'red' },
        { date: '12 Jan 2026', status: 'Taken by Admin', icon: FaClock, color: 'yellow' },
        { date: '15 Jan 2026', status: 'Closed', icon: FaCheckCircle, color: 'green' }
      ],
      userFeedback: {
        rating: 4,
        comment: 'Good service'
      }
    }
  ];

  const renderStars = (rating) => {
    return Array.from({ length: 5 }, (_, i) => (
      <FaStar key={i} className={`text-xl ${i < rating ? 'text-yellow-400' : 'text-gray-300'}`} />
    ));
  };

  // Generate dummy password based on zone
  const generatePassword = (zoneName) => {
    const zoneNum = zoneName.match(/\d+/)?.[0] || '1';
    return `Zone${zoneNum}Admin@2025`;
  };

  const adminPassword = generatePassword(admin.zone);

  const [showPassword, setShowPassword] = useState(false);

  const displayedPassword = showPassword ? adminPassword : '•'.repeat(Math.max(8, adminPassword.length));

  const toggleShowPassword = () => setShowPassword(s => !s);

  const copyToClipboard = (text, label) => {
    navigator.clipboard.writeText(text).then(() => {
      alert(`${label} copied to clipboard!`);
    }).catch(() => {
      alert('Unable to copy to clipboard');
    });
  };

  return (
    <div className="mt-8 px-4 md:px-6">
      {/* Back Button */}
      <button
        onClick={() => navigate('/city-admin')}
        className="mb-6 flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200 transition-colors"
      >
        <FaArrowLeft />
        Back to City Dashboard
      </button>

      {/* Admin Info Banner */}
      <div className="mb-8 p-6 rounded-xl bg-gradient-to-r from-purple-600 to-indigo-600 dark:from-purple-700 dark:to-indigo-700 shadow-lg">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="p-4 bg-white/20 backdrop-blur-sm rounded-full">
              <FaUserShield className="text-3xl text-white" />
            </div>
            <div>
              <h2 className="text-2xl md:text-3xl font-bold text-white mb-2">
                {admin.name}
              </h2>
              <div className="flex items-center gap-2 flex-wrap">
                <span className="px-4 py-1.5 bg-white/20 backdrop-blur-sm rounded-full text-white font-semibold text-sm">
                  📍 {admin.zone}
                </span>
                <span className="px-4 py-1.5 bg-white/20 backdrop-blur-sm rounded-full text-white font-semibold text-sm">
                  📧 {admin.email}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Admin Credentials Section */}
     

      {/* Performance Summary */}
      <div className="mb-8">
        <h2 className="text-2xl font-bold text-gray-800 dark:text-white mb-4 flex items-center gap-2">
          📊 Performance Summary
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <div className="bg-white dark:bg-secondary-dark-bg rounded-xl shadow-lg p-6 border-l-4 border-blue-500">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-600 dark:text-gray-400 text-sm mb-1">Total Complaints Assigned</p>
                <p className="text-3xl font-bold text-blue-600">{admin.totalComplaints}</p>
              </div>
              <div className="p-4 bg-blue-100 dark:bg-blue-900/20 rounded-full">
                <FaClipboardCheck className="text-2xl text-blue-600" />
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-secondary-dark-bg rounded-xl shadow-lg p-6 border-l-4 border-green-500">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-600 dark:text-gray-400 text-sm mb-1">Complaints Resolved</p>
                <p className="text-3xl font-bold text-green-600">{admin.resolved}</p>
              </div>
              <div className="p-4 bg-green-100 dark:bg-green-900/20 rounded-full">
                <FaCheckCircle className="text-2xl text-green-600" />
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-secondary-dark-bg rounded-xl shadow-lg p-6 border-l-4 border-orange-500">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-600 dark:text-gray-400 text-sm mb-1">Avg Resolution Time</p>
                <p className="text-2xl font-bold text-orange-600">{admin.avgResolutionTime}</p>
              </div>
              <div className="p-4 bg-orange-100 dark:bg-orange-900/20 rounded-full">
                <FaClock className="text-2xl text-orange-600" />
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-secondary-dark-bg rounded-xl shadow-lg p-6 border-l-4 border-yellow-500">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-600 dark:text-gray-400 text-sm mb-1">Overall User Feedback</p>
                <div className="flex gap-1 mt-2">
                  {renderStars(admin.rating)}
                </div>
                <p className="text-sm font-semibold text-yellow-600 mt-1">
                  {admin.rating === 5 ? 'Very Good' : admin.rating >= 4 ? 'Good' : 'Average'}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* All Complaints Section */}
      <div className="mb-8">
        <h2 className="text-2xl font-bold text-gray-800 dark:text-white mb-6 flex items-center gap-2">
          <FaClipboardCheck style={{ color: currentColor }} />
          All Complaints ({complaintsData.length})
        </h2>

        <div className="space-y-6">
          {complaintsData.map((complaint, idx) => (
            <div key={complaint.id} className="bg-white dark:bg-secondary-dark-bg rounded-xl shadow-lg p-6">
              {/* Complaint Header */}
              <div className="mb-4 pb-4 border-b border-gray-200 dark:border-gray-700">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-lg font-bold text-gray-800 dark:text-white">
                    Complaint #{complaint.id}
                  </h3>
                  <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
                    complaint.userFeedback.rating === 5 
                      ? 'bg-green-100 dark:bg-green-900/20 text-green-700 dark:text-green-300'
                      : complaint.userFeedback.rating >= 4
                      ? 'bg-blue-100 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300'
                      : 'bg-yellow-100 dark:bg-yellow-900/20 text-yellow-700 dark:text-yellow-300'
                  }`}>
                    {complaint.userFeedback.rating === 5 ? '⭐ Excellent' : complaint.userFeedback.rating >= 4 ? '⭐ Good' : '⭐ Average'}
                  </span>
                </div>
              </div>

              {/* Complaint Details */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div className="space-y-3">
                  <div className="flex items-start gap-3">
                    <div className="p-2 rounded-lg" style={{ backgroundColor: `${currentColor}20` }}>
                      <FaUserShield style={{ color: currentColor }} />
                    </div>
                    <div>
                      <p className="text-xs text-gray-600 dark:text-gray-400">Submitted by</p>
                      <p className="font-semibold text-gray-800 dark:text-white">{complaint.submittedBy}</p>
                    </div>
                  </div>

                  <div className="flex items-start gap-3">
                    <div className="p-2 rounded-lg" style={{ backgroundColor: `${currentColor}20` }}>
                      <FaPhone style={{ color: currentColor }} />
                    </div>
                    <div>
                      <p className="text-xs text-gray-600 dark:text-gray-400">Phone</p>
                      <p className="font-semibold text-gray-800 dark:text-white">{complaint.phone}</p>
                    </div>
                  </div>
                </div>

                <div className="space-y-3">
                  <div className="flex items-start gap-3">
                    <div className="p-2 rounded-lg" style={{ backgroundColor: `${currentColor}20` }}>
                      <FaMapMarkerAlt style={{ color: currentColor }} />
                    </div>
                    <div>
                      <p className="text-xs text-gray-600 dark:text-gray-400">Zone</p>
                      <p className="font-semibold text-gray-800 dark:text-white">{complaint.zone}</p>
                    </div>
                  </div>

                  <div className="flex items-start gap-3">
                    <div className="p-2 rounded-lg" style={{ backgroundColor: `${currentColor}20` }}>
                      <FaMapMarkerAlt style={{ color: currentColor }} />
                    </div>
                    <div>
                      <p className="text-xs text-gray-600 dark:text-gray-400">Address</p>
                      <p className="font-semibold text-gray-800 dark:text-white">{complaint.address}</p>
                    </div>
                  </div>
                </div>
              </div>

              <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg mb-6">
                <p className="text-xs text-gray-600 dark:text-gray-400 mb-2">Description</p>
                <p className="text-gray-800 dark:text-white font-medium">{complaint.description}</p>
              </div>

              {/* Timeline */}
              <div className="mb-6">
                <h3 className="text-sm font-bold text-gray-700 dark:text-gray-300 mb-4 flex items-center gap-2">
                  <MdTimeline style={{ color: currentColor }} />
                  Timeline
                </h3>
                <div className="relative">
                  <div className="absolute top-8 left-0 right-0 h-1 bg-gray-200 dark:bg-gray-700 hidden md:block"></div>
                  <div 
                    className="absolute top-8 left-0 h-1 hidden md:block" 
                    style={{ 
                      width: '100%',
                      background: 'linear-gradient(to right, #ef4444, #eab308, #22c55e)'
                    }}
                  ></div>

                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 relative">
                    {complaint.timeline.map((step, index) => {
                      const Icon = step.icon;
                      const colorMap = {
                        red: { bg: 'bg-red-100 dark:bg-red-900/20', text: 'text-red-600', border: 'border-red-500' },
                        yellow: { bg: 'bg-yellow-100 dark:bg-yellow-900/20', text: 'text-yellow-600', border: 'border-yellow-500' },
                        green: { bg: 'bg-green-100 dark:bg-green-900/20', text: 'text-green-600', border: 'border-green-500' }
                      };
                      const colors = colorMap[step.color];

                      return (
                        <div key={index} className="flex flex-col items-center text-center relative z-10">
                          <div className={`w-14 h-14 rounded-full ${colors.bg} flex items-center justify-center mb-2 border-4 ${colors.border} bg-white dark:bg-gray-800`}>
                            <Icon className={`text-xl ${colors.text}`} />
                          </div>
                          <p className="font-bold text-gray-800 dark:text-white text-sm mb-1">{step.status}</p>
                          <p className="text-xs text-gray-600 dark:text-gray-400">{step.date}</p>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>

              {/* User Feedback */}
              <div className="bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg p-4 border border-green-200 dark:border-green-800">
                <h3 className="text-sm font-bold text-gray-700 dark:text-gray-300 mb-3 flex items-center gap-2">
                  <FaStar className="text-yellow-500" />
                  User Feedback
                </h3>
                <div className="flex items-center justify-between">
                  <div className="flex gap-1">
                    {renderStars(complaint.userFeedback.rating)}
                  </div>
                  <div>
                    <span className="px-3 py-1 bg-green-100 dark:bg-green-900/20 text-green-700 dark:text-green-300 rounded-full font-bold text-sm">
                      {complaint.userFeedback.rating}/5
                    </span>
                  </div>
                </div>
                <p className="mt-3 text-sm text-gray-800 dark:text-white italic">
                  "{complaint.userFeedback.comment}"
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Info Note */}
      <div className="mt-8 bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 rounded-lg p-4">
        <p className="text-sm text-purple-800 dark:text-purple-200">
          <strong>Purpose:</strong> This dashboard demonstrates the City Admin's ability to monitor zone admin efficiency, 
          track complaint resolution times, evaluate feedback quality, and ensure accountability across all zones.
        </p>
      </div>
    </div>
  );
};

export default ZoneAdminPerformance;
