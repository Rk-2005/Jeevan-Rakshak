import React, { useState, useEffect } from 'react';
import { useLanguage } from '../contexts/LanguageContext';
import { useStateContext } from '../contexts/ContextProvider';
import { Button } from '../components';
import { FaExclamationTriangle, FaMapMarkerAlt, FaChartLine, FaThermometerHalf } from 'react-icons/fa';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

const OutbreakRisk = () => {
  const { t } = useLanguage();
  const { currentColor } = useStateContext();
  const [riskData, setRiskData] = useState(null);
  const [loading, setIsLoading] = useState(true);

  // Mock data for demonstration
  const mockRiskData = {
    overallRisk: 'High',
    riskScore: 85,
    hotspots: [
      { id: 1, village: 'Village A', risk: 'High', cases: 12, waterQuality: 'Poor', coordinates: [28.6139, 77.2090] },
      { id: 2, village: 'Village B', risk: 'Medium', cases: 6, waterQuality: 'Fair', coordinates: [28.6140, 77.2091] },
      { id: 3, village: 'Village C', risk: 'Low', cases: 2, waterQuality: 'Good', coordinates: [28.6141, 77.2092] },
      { id: 4, village: 'Village D', risk: 'High', cases: 15, waterQuality: 'Poor', coordinates: [28.6142, 77.2093] },
      { id: 5, village: 'Village E', risk: 'Medium', cases: 4, waterQuality: 'Fair', coordinates: [28.6143, 77.2094] }
    ],
    riskFactors: {
      waterQuality: 0.7,
      reportedCases: 0.8,
      populationDensity: 0.6,
      seasonalFactors: 0.9
    },
    trendData: [
      { month: 'Jan', risk: 45, cases: 5 },
      { month: 'Feb', risk: 52, cases: 8 },
      { month: 'Mar', risk: 38, cases: 3 },
      { month: 'Apr', risk: 65, cases: 12 },
      { month: 'May', risk: 78, cases: 18 },
      { month: 'Jun', risk: 85, cases: 25 }
    ],
    recommendations: [
      'Immediate water quality testing required in high-risk areas',
      'Deploy additional health monitoring teams',
      'Implement emergency water treatment measures',
      'Increase public awareness campaigns',
      'Consider temporary water supply alternatives'
    ]
  };

  useEffect(() => {
    // Simulate API call
    const fetchRiskData = async () => {
      setIsLoading(true);
      await new Promise(resolve => setTimeout(resolve, 1500));
      setRiskData(mockRiskData);
      setIsLoading(false);
    };

    fetchRiskData();
  }, []);

  const getRiskColor = (risk) => {
    switch (risk) {
      case 'High': return '#ef4444';
      case 'Medium': return '#f59e0b';
      case 'Low': return '#10b981';
      default: return '#6b7280';
    }
  };
  const [show,setshow]=useState(false);

  const getRiskIcon = (risk) => {
    switch (risk) {
      case 'High': return '🔴';
      case 'Medium': return '🟡';
      case 'Low': return '🟢';
      default: return '⚪';
    }
  };

  const pieData = [
    { name: 'High Risk', value: mockRiskData?.hotspots.filter(h => h.risk === 'High').length || 0, color: '#ef4444' },
    { name: 'Medium Risk', value: mockRiskData?.hotspots.filter(h => h.risk === 'Medium').length || 0, color: '#f59e0b' },
    { name: 'Low Risk', value: mockRiskData?.hotspots.filter(h => h.risk === 'Low').length || 0, color: '#10b981' }
  ];

  if (loading) {
    return (
      <div className="mt-24 px-4 md:px-6">
        <div className="flex justify-center items-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2" style={{ borderColor: currentColor }}></div>
        </div>
      </div>
    );
  }
const showsomething=()=>{
  setshow(true);
}
  return (
    <div className="mt-24 px-4 md:px-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-800 dark:text-white mb-4">
            <FaExclamationTriangle className="inline-block mr-3" style={{ color: currentColor }} />
            {t('outbreakRisk')} Analysis
          </h1>
          <p className="text-gray-600 dark:text-gray-300">
            AI-powered outbreak risk assessment and hotspot identification
          </p>
        </div>

        {/* Risk Overview Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          {/* Overall Risk Score */}
      

          {/* Active Hotspots */}
          <div className="bg-white dark:bg-secondary-dark-bg rounded-xl shadow-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-800 dark:text-white">Active Hotspots</h3>
              <FaMapMarkerAlt className="text-2xl" style={{ color: currentColor }} />
            </div>
            <div className="text-center">
              <div className="text-4xl font-bold mb-2" style={{ color: currentColor }}>
                {riskData.hotspots.length}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-300">
                Villages under monitoring
              </div>
            </div>
          </div>

          {/* Risk Trend */}
        
        </div>

        {/* Charts Row */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* Risk Trend Chart */}
          <div className="bg-white dark:bg-secondary-dark-bg rounded-xl shadow-lg p-6">
            <h3 className="text-lg font-semibold text-gray-800 dark:text-white mb-4">Risk Trend Over Time</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={riskData.trendData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" />
                <YAxis />
                <Tooltip />
                <Line 
                  type="monotone" 
                  dataKey="risk" 
                  stroke={currentColor} 
                  strokeWidth={3}
                  dot={{ fill: currentColor, strokeWidth: 2, r: 6 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Risk Distribution Pie Chart */}
        
        </div>

        {/* Hotspots Table */}
        <div className="bg-white dark:bg-secondary-dark-bg rounded-xl shadow-lg p-6 mb-8">
          <h3 className="text-lg font-semibold text-gray-800 dark:text-white mb-4">Risk Hotspots</h3>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left py-3 px-4 font-medium text-gray-700 dark:text-gray-300">Village</th>
                  <th className="text-left py-3 px-4 font-medium text-gray-700 dark:text-gray-300">Risk Level</th>
                  <th className="text-left py-3 px-4 font-medium text-gray-700 dark:text-gray-300">Cases</th>
                  <th className="text-left py-3 px-4 font-medium text-gray-700 dark:text-gray-300">Water Quality</th>
                  <th className="text-left py-3 px-4 font-medium text-gray-700 dark:text-gray-300">Actions</th>
                </tr>
              </thead>
              <tbody>
                {riskData.hotspots.map((hotspot) => (
                  <tr key={hotspot.id} className="border-b border-gray-100 dark:border-gray-800">
                    <td className="py-3 px-4 text-gray-800 dark:text-white">{hotspot.village}</td>
                    <td className="py-3 px-4">
                      <span 
                        className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium"
                        style={{ 
                          backgroundColor: `${getRiskColor(hotspot.risk)}20`,
                          color: getRiskColor(hotspot.risk)
                        }}
                      >
                        {getRiskIcon(hotspot.risk)} {hotspot.risk}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-gray-800 dark:text-white">{hotspot.cases}</td>
                    <td className="py-3 px-4">
                      <span 
                        className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                          hotspot.waterQuality === 'Good' ? 'bg-green-100 text-green-800' :
                          hotspot.waterQuality === 'Fair' ? 'bg-yellow-100 text-yellow-800' :
                          'bg-red-100 text-red-800'
                        }`}
                      >
                        {hotspot.waterQuality}
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      <Button
                        color="white"
                        bgColor={currentColor}
                        text="View Details"
                        borderRadius="6px"
                        size="sm" 
                        onClick={showsomething}
                      />
                    </td>
                    {show && <div className="m-4">Coming soon</div>}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Recommendations */}
       
      </div>
    </div>
  );
};

export default OutbreakRisk;
