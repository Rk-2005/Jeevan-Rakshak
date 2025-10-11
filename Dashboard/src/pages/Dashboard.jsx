import React, { useEffect, useState } from 'react';
import { get, ref } from 'firebase/database';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { FaFilter, FaDownload, FaChartBar, FaExclamationTriangle, FaCheckCircle } from 'react-icons/fa';

import { Button, SparkLine } from '../components';
import { useStateContext } from '../contexts/ContextProvider';
import { useLanguage } from '../contexts/LanguageContext';
import product9 from '../data/JR3.png';
import { SparklineAreaData } from '../data/dummy';
import { database } from '../firebaseConfig';
import TestComponent from '../TestComponent';

const Dashboard = () => {
  const { currentColor } = useStateContext();
  const { t } = useLanguage();
  const [waterSavedValue, setWaterSavedValue] = useState('');
  const [activeSensors, setActiveSensors] = useState('38');
  const [fraudsDetected, setFraudsDetected] = useState('1');
  const [leaksDetected, setLeaksDetected] = useState('7');
  const [reportedComplaints, setReportedComplaints] = useState('4');
  const [filters, setFilters] = useState({
    village: 'all',
    district: 'all',
    timePeriod: '30days',
  });
  const [showFilters, setShowFilters] = useState(false);

  // Mock data for enhanced dashboard
  const kpiData = {
    safeWaterSources: 75,
    unsafeWaterSources: 25,
    casesPer1000: 12.5,
    activeComplaints: 8,
    resolvedComplaints: 15,
    totalPopulation: 50000,
  };

  const waterQualityTrend = [
    { month: 'Jan', safe: 65, unsafe: 35 },
    { month: 'Feb', safe: 68, unsafe: 32 },
    { month: 'Mar', safe: 72, unsafe: 28 },
    { month: 'Apr', safe: 70, unsafe: 30 },
    { month: 'May', safe: 75, unsafe: 25 },
    { month: 'Jun', safe: 78, unsafe: 22 },
  ];

  const complaintStatusData = [
    { name: 'Resolved', value: 15, color: '#10b981' },
    { name: 'In Progress', value: 8, color: '#f59e0b' },
    { name: 'Pending', value: 4, color: '#ef4444' },
  ];

  const villageData = [
    { name: 'Village A', cases: 12, waterQuality: 'Poor', risk: 'High' },
    { name: 'Village B', cases: 6, waterQuality: 'Fair', risk: 'Medium' },
    { name: 'Village C', cases: 2, waterQuality: 'Good', risk: 'Low' },
    { name: 'Village D', cases: 15, waterQuality: 'Poor', risk: 'High' },
    { name: 'Village E', cases: 4, waterQuality: 'Fair', risk: 'Medium' },
  ];
  const [show,setshow]=useState(false);

  const download=()=>{
    setshow(true);
  }

  useEffect(() => {
    const fetchValues = async () => {
      try {
        const waterSavedRef = ref(database, 'WaterSaved');
        const activeSensorsRef = ref(database, 'activeSensors');
        const fraudsDetectedRef = ref(database, 'fraudsDetected');
        const leaksDetectedRef = ref(database, 'leaksDetected');
        const reportedComplaintsRef = ref(database, 'reportedComplaints');

        const snapshot1 = await get(waterSavedRef);
        const snapshot2 = await get(activeSensorsRef);
        const snapshot3 = await get(fraudsDetectedRef);
        const snapshot4 = await get(leaksDetectedRef);
        const snapshot5 = await get(reportedComplaintsRef);

        if (snapshot1.exists()) {
          setWaterSavedValue(snapshot1.val());
        }
        if (snapshot2.exists()) {
          setActiveSensors(snapshot2.val());
        }
        if (snapshot3.exists()) {
          setFraudsDetected(snapshot3.val());
        }
        if (snapshot4.exists()) {
          setLeaksDetected(snapshot4.val());
        }
        if (snapshot5.exists()) {
          setReportedComplaints(snapshot5.val());
        }
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    };

    fetchValues();
  }, []);

  const handleExport = (format) => {
    // Mock export functionality
    const data = {
      waterSaved: waterSavedValue,
      activeSensors,
      fraudsDetected,
      leaksDetected,
      reportedComplaints,
      kpiData,
      waterQualityTrend,
      complaintStatusData,
      villageData,
      exportDate: new Date().toISOString()
    };

    if (format === 'pdf') {
      // Mock PDF export
      console.log('Exporting to PDF:', data);
      alert('Coming soon');
    } else if (format === 'excel') {
      // Mock Excel export
      console.log('Exporting to Excel:', data);
      alert('Coming soon');
    }
  };

  const handleFilterChange = (filterType, value) => {
    setFilters((prev) => ({
      ...prev,
      [filterType]: value,
    }));
  };

  return (
    <div className="mt-8 px-4 md:px-6">
      <TestComponent />
      {/* Header with Filters and Export */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-1">
        <div className='mb-10 pb-10'>
          <h1 className="text-3xl font-bold text-slate-800 dark:text-white mb-2">
            Community Health Monitoring Dashboard
          </h1>
          <p className="text-gray-600 dark:text-gray-300">
            Real-time monitoring and analysis of water quality and health data
          </p>
        </div>
        <div className="flex items-center space-x-4   mb-10 pb-10">
          <Button
            color="white"
            bgColor={currentColor}
            text="Filters"
            borderRadius="8px"
            size="md"
            icon={<FaFilter />}
            onClick={() => setShowFilters(!showFilters)}
          />
          <div className="relative group">
            <Button
              color="white"
              bgColor={currentColor}
              text="Export"
              borderRadius="8px"
              size="md"
              icon={<FaDownload />}
            />
            <div className="absolute right-0 mt-2 w-48 bg-white dark:bg-secondary-dark-bg rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 z-10 hidden group-hover:block">
              <buttons
                type="button"
                onClick={() => handleExport('pdf')}
                className="w-full text-left px-4 py-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-t-lg"
              >
                Export as PDF
              </buttons>
              <button
                type="button"
                onClick={() => handleExport('excel')}
                className="w-full text-left px-4 py-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-b-lg"
              >
                Export as Excel
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Filters Panel */}
      {showFilters && (
        <div className="bg-white dark:bg-secondary-dark-bg rounded-xl shadow-lg p-6 mb-8">
          <h3 className="text-lg font-semibold text-gray-800 dark:text-white mb-4">Filters</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Village
              </label>
              <select
                value={filters.village}
                onChange={(e) => handleFilterChange('village', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
              >
                <option value="all">All Villages</option>
                <option value="village-a">Village A</option>
                <option value="village-b">Village B</option>
                <option value="village-c">Village C</option>
                <option value="village-d">Village D</option>
                <option value="village-e">Village E</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                District
              </label>
              <select
                value={filters.district}
                onChange={(e) => handleFilterChange('district', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
              >
                <option value="all">All Districts</option>
                <option value="district-1">District 1</option>
                <option value="district-2">District 2</option>
                <option value="district-3">District 3</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Time Period
              </label>
              <select
                value={filters.timePeriod}
                onChange={(e) => handleFilterChange('timePeriod', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
              >
                <option value="7days">Last 7 days</option>
                <option value="30days">Last 30 days</option>
                <option value="90days">Last 90 days</option>
                <option value="1year">Last year</option>
              </select>
            </div>
          </div>
        </div>
      )}

      {/* KPI Cards */}
    

      {/* Hero Card and Stats */}
      <div className="flex flex-col lg:flex-row gap-4 justify-center">
        {/* Hero Card */}
        <div className="bg-white dark:text-gray-200 dark:bg-secondary-dark-bg rounded-xl w-full lg:w-80 p-6 md:p-8 bg-gradient-to-r from-blue-50 to-blue-100 shadow-md">
          <div className="flex justify-between items-center">
            <div>
              <p className="font-bold text-gray-500">{t('waterSaved')}</p>
              <p className="text-2xl font-semibold">{waterSavedValue} liters</p>
            </div>
            <button
              type="button"
              style={{ backgroundColor: currentColor }}
              className="text-2xl opacity-90 text-white hover:shadow-lg rounded-full p-3 transition-all duration-200"
            >
              💧
            </button>
          </div>
          <div className="mt-6">
            <Button
              color="white"
              bgColor={currentColor}
              text={t('downloadReport')}
              borderRadius="10px"
              size="md"
              onClick={download}
            />
            {show && <div className='text-blue-400 font-bold text-xl mt-2 ml-1'>Coming soon</div>}
          </div>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 flex-1">
          <div className="bg-white dark:text-gray-200 dark:bg-secondary-dark-bg p-4 rounded-xl shadow-sm hover:shadow-md transition-shadow duration-200">
            <p className="text-2xl font-semibold">{leaksDetected}</p>
            <p className="text-sm text-gray-500 mt-1">{t('unsafeWaterSources')}</p>
          </div>

          <div className="bg-white dark:text-gray-200 dark:bg-secondary-dark-bg p-4 rounded-xl shadow-sm hover:shadow-md transition-shadow duration-200">
            <p className="text-2xl font-semibold">{fraudsDetected}</p>
            <p className="text-sm text-gray-500 mt-1">{t('detectedOutbreakRisks')}</p>
          </div>

          <div className="bg-white dark:text-gray-200 dark:bg-secondary-dark-bg p-4 rounded-xl shadow-sm hover:shadow-md transition-shadow duration-200">
            <p className="text-2xl font-semibold">{reportedComplaints}</p>
            <p className="text-sm text-gray-500 mt-1">{t('reportedComplaints')}</p>
          </div>

          <div className="bg-white dark:text-gray-200 dark:bg-secondary-dark-bg p-4 rounded-xl shadow-sm hover:shadow-md transition-shadow duration-200">
            <p className="text-2xl font-semibold">{activeSensors}</p>
            <p className="text-sm text-gray-500 mt-1">{t('activeSensors')}</p>
          </div>
        </div>
      </div>

      {/* Power BI Section */}
      <div className="mt-8">
        <div className="bg-blue-50 text-blue-800 px-4 py-3 rounded-lg mb-4 border-l-4 border-blue-500 font-medium text-center">
          Live Detections are on! For help with any technical issue, please contact us at 8265096155
        </div>
        <div className="rounded-xl overflow-hidden shadow-lg">
          <iframe
            title="Report Section"
            className="w-full h-[500px] md:h-[800px]"
            src="https://app.powerbi.com/view?r=eyJrIjoiNjI0NDViMTgtM2QzMS00YzYzLTk4MDYtZWQyZmQzY2Y3ODg2IiwidCI6IjNmMzFkNjNkLWVkYzMtNDEzZS04N2U0LTQyMGU1M2ZkZDYyZiJ9"
            frameBorder="0"
            allowFullScreen
          />
        </div>
      </div>

      {/* Sparkline and App Card Section */}
      <div className="mt-8 flex flex-col lg:flex-row gap-6">
        {/* Sparkline Card */}
        <div 
          className="rounded-xl p-6 w-full lg:w-2/3 shadow-lg"
          style={{ backgroundColor: currentColor }}
        >
          <div className="flex flex-col md:flex-row justify-between items-start md:items-center">
            <div>
              <p className="font-semibold text-white text-2xl">{t('outbreaksDetections')}</p>
              <p className="text-gray-200 mt-2">{t('monthlyStatistics')}</p>
            </div>
            <div className="mt-4 md:mt-0">
              <p className="text-2xl text-white font-semibold">930 liters</p>
              <p className="text-gray-200">{t('monthlyDetects')}</p>
            </div>
          </div>
          <div className="mt-6">
            <SparkLine
              currentColor={currentColor}
              id="column-sparkLine"
              height="100px"
              type="Column"
              data={SparklineAreaData}
              width="100%"
              color="rgb(242, 252, 253)"
            />
          </div>
        </div>

        {/* App Download Card */}
        <div className="bg-white dark:text-gray-200 dark:bg-secondary-dark-bg rounded-xl p-6 w-full lg:w-1/3 shadow-lg flex flex-col">
          <img 
            className="w-full h-auto rounded-lg object-cover" 
            src={product9} 
            alt="App Screenshot" 
          />
          <div className="mt-6 flex-1 flex flex-col">
            <p className="font-semibold text-lg">{t('downloadApp')}</p>
            <p className="text-gray-500 text-sm">By Team Jeevan-Rakshak</p>
            <p className="mt-4 text-gray-600 dark:text-gray-300 text-sm flex-1">
              {t('appDescription')}
            </p>
            <Button
              color="white"
              bgColor={currentColor}
              text={t('downloadApp')}
              borderRadius="8px"
              size="md"
              className="mt-6 w-full"
            />
          </div>
        </div>
      </div>

      {/* Enhanced Charts Section */}
      <div className="mt-8 grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Water Quality Trend Chart */}
        <div className="bg-white dark:bg-secondary-dark-bg rounded-xl shadow-lg p-6">
          <h3 className="text-lg font-semibold text-gray-800 dark:text-white mb-4">
            Water Quality Trend
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={waterQualityTrend}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="month" />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="safe" stroke="#10b981" strokeWidth={3} name="Safe Sources %" />
              <Line type="monotone" dataKey="unsafe" stroke="#ef4444" strokeWidth={3} name="Unsafe Sources %" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Complaint Status Distribution */}
        <div className="bg-white dark:bg-secondary-dark-bg rounded-xl shadow-lg p-6">
          <h3 className="text-lg font-semibold text-gray-800 dark:text-white mb-4">
            Complaint Status Distribution
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={complaintStatusData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {complaintStatusData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Village Performance Table */}
      <div className="mt-8 bg-white dark:bg-secondary-dark-bg rounded-xl shadow-lg p-6">
        <h3 className="text-lg font-semibold text-gray-800 dark:text-white mb-4">
          Village Performance Overview
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-200 dark:border-gray-700">
                <th className="text-left py-3 px-4 font-medium text-gray-700 dark:text-gray-300">Village</th>
                <th className="text-left py-3 px-4 font-medium text-gray-700 dark:text-gray-300">Health Cases</th>
                <th className="text-left py-3 px-4 font-medium text-gray-700 dark:text-gray-300">Water Quality</th>
                <th className="text-left py-3 px-4 font-medium text-gray-700 dark:text-gray-300">Risk Level</th>
                <th className="text-left py-3 px-4 font-medium text-gray-700 dark:text-gray-300">Status</th>
              </tr>
            </thead>
            <tbody>
              {villageData.map((village, index) => (
                <tr key={index} className="border-b border-gray-100 dark:border-gray-800">
                  <td className="py-3 px-4 text-gray-800 dark:text-white font-medium">{village.name}</td>
                  <td className="py-3 px-4 text-gray-800 dark:text-white">{village.cases}</td>
                  <td className="py-3 px-4">
                    {(() => {
                      const getWaterQualityClass = (quality) => {
                        if (quality === 'Good') return 'bg-green-100 text-green-800';
                        if (quality === 'Fair') return 'bg-yellow-100 text-yellow-800';
                        return 'bg-red-100 text-red-800';
                      };
                      return (
                        <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${getWaterQualityClass(village.waterQuality)}`}>
                          {village.waterQuality}
                        </span>
                      );
                    })()}
                  </td>
                  <td className="py-3 px-4">
                    {(() => {
                      const getRiskClass = (risk) => {
                        if (risk === 'Low') return 'bg-green-100 text-green-800';
                        if (risk === 'Medium') return 'bg-yellow-100 text-yellow-800';
                        return 'bg-red-100 text-red-800';
                      };
                      return (
                        <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${getRiskClass(village.risk)}`}>
                          {village.risk}
                        </span>
                      );
                    })()}
                  </td>
                  <td className="py-3 px-4">
                    {(() => {
                      const getStatusClass = (risk) => {
                        if (risk === 'Low') return 'bg-green-100 text-green-800';
                        if (risk === 'Medium') return 'bg-yellow-100 text-yellow-800';
                        return 'bg-red-100 text-red-800';
                      };
                      const getStatusText = (risk) => {
                        if (risk === 'Low') return '✓ Safe';
                        if (risk === 'Medium') return '⚠ Monitor';
                        return '🚨 Alert';
                      };
                      return (
                        <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${getStatusClass(village.risk)}`}>
                          {getStatusText(village.risk)}
                        </span>
                      );
                    })()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
