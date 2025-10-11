import React, { useState, useEffect } from 'react';
import { useLanguage } from '../contexts/LanguageContext';
import { useStateContext } from '../contexts/ContextProvider';
import { Button } from '../components';
import { FaTint, FaThermometerHalf, FaFlask, FaExclamationTriangle, FaCheckCircle, FaTimesCircle } from 'react-icons/fa';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';

const WaterQualitySensors = () => {
  const { t } = useLanguage();
  const { currentColor } = useStateContext();
  const [sensors, setSensors] = useState([]);
  const [selectedSensor, setSelectedSensor] = useState(null);
  const [loading, setIsLoading] = useState(true);

  // Mock sensor data
  const mockSensors = [
    {
      id: 1,
      name: 'Sensor Node A',
      location: 'Village A - Main Water Source',
      status: 'active',
      coordinates: [28.6139, 77.2090],
      parameters: {
        ph: { value: 6.8, status: 'warning', min: 6.5, max: 8.5, unit: 'pH' },
        turbidity: { value: 4.2, status: 'good', min: 0, max: 5, unit: 'NTU' },
        bacterialCount: { value: 150, status: 'warning', min: 0, max: 100, unit: 'CFU/100ml' },
        chlorineLevels: { value: 0.8, status: 'good', min: 0.2, max: 4, unit: 'mg/L' },
        temperature: { value: 25.5, status: 'good', min: 15, max: 30, unit: '°C' },
        dissolvedOxygen: { value: 7.2, status: 'good', min: 5, max: 10, unit: 'mg/L' }
      },
      lastUpdate: '2024-01-15T10:30:00Z',
      historicalData: [
        { time: '00:00', ph: 6.9, turbidity: 3.8, bacterialCount: 120, chlorineLevels: 0.9, temperature: 24.8, dissolvedOxygen: 7.5 },
        { time: '04:00', ph: 6.7, turbidity: 4.1, bacterialCount: 135, chlorineLevels: 0.8, temperature: 25.1, dissolvedOxygen: 7.3 },
        { time: '08:00', ph: 6.8, turbidity: 4.2, bacterialCount: 150, chlorineLevels: 0.8, temperature: 25.5, dissolvedOxygen: 7.2 },
        { time: '12:00', ph: 6.9, turbidity: 4.0, bacterialCount: 145, chlorineLevels: 0.7, temperature: 26.2, dissolvedOxygen: 7.0 },
        { time: '16:00', ph: 7.0, turbidity: 3.9, bacterialCount: 140, chlorineLevels: 0.8, temperature: 26.8, dissolvedOxygen: 6.8 },
        { time: '20:00', ph: 6.8, turbidity: 4.3, bacterialCount: 155, chlorineLevels: 0.9, temperature: 25.9, dissolvedOxygen: 7.1 }
      ]
    },
    {
      id: 2,
      name: 'Sensor Node B',
      location: 'Village B - Treatment Plant',
      status: 'active',
      coordinates: [28.6140, 77.2091],
      parameters: {
        ph: { value: 7.2, status: 'good', min: 6.5, max: 8.5, unit: 'pH' },
        turbidity: { value: 2.1, status: 'good', min: 0, max: 5, unit: 'NTU' },
        bacterialCount: { value: 45, status: 'good', min: 0, max: 100, unit: 'CFU/100ml' },
        chlorineLevels: { value: 1.2, status: 'good', min: 0.2, max: 4, unit: 'mg/L' },
        temperature: { value: 24.8, status: 'good', min: 15, max: 30, unit: '°C' },
        dissolvedOxygen: { value: 8.1, status: 'good', min: 5, max: 10, unit: 'mg/L' }
      },
      lastUpdate: '2024-01-15T10:25:00Z',
      historicalData: [
        { time: '00:00', ph: 7.1, turbidity: 2.3, bacterialCount: 50, chlorineLevels: 1.3, temperature: 24.5, dissolvedOxygen: 8.3 },
        { time: '04:00', ph: 7.2, turbidity: 2.2, bacterialCount: 48, chlorineLevels: 1.2, temperature: 24.7, dissolvedOxygen: 8.2 },
        { time: '08:00', ph: 7.2, turbidity: 2.1, bacterialCount: 45, chlorineLevels: 1.2, temperature: 24.8, dissolvedOxygen: 8.1 },
        { time: '12:00', ph: 7.3, turbidity: 2.0, bacterialCount: 42, chlorineLevels: 1.1, temperature: 25.2, dissolvedOxygen: 7.9 },
        { time: '16:00', ph: 7.1, turbidity: 2.2, bacterialCount: 47, chlorineLevels: 1.3, temperature: 25.8, dissolvedOxygen: 7.7 },
        { time: '20:00', ph: 7.2, turbidity: 2.1, bacterialCount: 45, chlorineLevels: 1.2, temperature: 25.1, dissolvedOxygen: 8.0 }
      ]
    },
    {
      id: 3,
      name: 'Sensor Node C',
      location: 'Village C - Distribution Point',
      status: 'warning',
      coordinates: [28.6141, 77.2092],
      parameters: {
        ph: { value: 5.8, status: 'danger', min: 6.5, max: 8.5, unit: 'pH' },
        turbidity: { value: 6.2, status: 'danger', min: 0, max: 5, unit: 'NTU' },
        bacterialCount: { value: 280, status: 'danger', min: 0, max: 100, unit: 'CFU/100ml' },
        chlorineLevels: { value: 0.1, status: 'danger', min: 0.2, max: 4, unit: 'mg/L' },
        temperature: { value: 28.5, status: 'warning', min: 15, max: 30, unit: '°C' },
        dissolvedOxygen: { value: 4.2, status: 'warning', min: 5, max: 10, unit: 'mg/L' }
      },
      lastUpdate: '2024-01-15T10:20:00Z',
      historicalData: [
        { time: '00:00', ph: 6.0, turbidity: 5.8, bacterialCount: 250, chlorineLevels: 0.2, temperature: 27.8, dissolvedOxygen: 4.5 },
        { time: '04:00', ph: 5.9, turbidity: 6.0, bacterialCount: 265, chlorineLevels: 0.1, temperature: 28.1, dissolvedOxygen: 4.3 },
        { time: '08:00', ph: 5.8, turbidity: 6.2, bacterialCount: 280, chlorineLevels: 0.1, temperature: 28.5, dissolvedOxygen: 4.2 },
        { time: '12:00', ph: 5.7, turbidity: 6.5, bacterialCount: 295, chlorineLevels: 0.1, temperature: 29.2, dissolvedOxygen: 4.0 },
        { time: '16:00', ph: 5.9, turbidity: 6.1, bacterialCount: 285, chlorineLevels: 0.2, temperature: 29.8, dissolvedOxygen: 3.8 },
        { time: '20:00', ph: 5.8, turbidity: 6.3, bacterialCount: 290, chlorineLevels: 0.1, temperature: 28.9, dissolvedOxygen: 4.1 }
      ]
    }
  ];

  useEffect(() => {
    const fetchSensors = async () => {
      setIsLoading(true);
      await new Promise(resolve => setTimeout(resolve, 1000));
      setSensors(mockSensors);
      setSelectedSensor(mockSensors[0]);
      setIsLoading(false);
    };

    fetchSensors();
  }, []);

  const getStatusIcon = (status) => {
    switch (status) {
      case 'good': return <FaCheckCircle className="text-green-500" />;
      case 'warning': return <FaExclamationTriangle className="text-yellow-500" />;
      case 'danger': return <FaTimesCircle className="text-red-500" />;
      default: return <FaTimesCircle className="text-gray-500" />;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'good': return 'text-green-600 bg-green-100';
      case 'warning': return 'text-yellow-600 bg-yellow-100';
      case 'danger': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getParameterIcon = (param) => {
    switch (param) {
      case 'ph': return <FaFlask className="text-blue-500" />;
      case 'turbidity': return <FaTint className="text-cyan-500" />;
      case 'bacterialCount': return <FaExclamationTriangle className="text-red-500" />;
      case 'chlorineLevels': return <FaFlask className="text-green-500" />;
      case 'temperature': return <FaThermometerHalf className="text-orange-500" />;
      case 'dissolvedOxygen': return <FaTint className="text-blue-600" />;
      default: return <FaFlask className="text-gray-500" />;
    }
  };

  if (loading) {
    return (
      <div className="mt-24 px-4 md:px-6">
        <div className="flex justify-center items-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2" style={{ borderColor: currentColor }}></div>
        </div>
      </div>
    );
  }

  return (
    <div className="mt-24 px-4 md:px-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-800 dark:text-white mb-4">
            <FaTint className="inline-block mr-1" style={{ color: currentColor }} />
            Water Quality Sensors
          </h1>
          <p className="text-gray-600 dark:text-gray-300">
            Real-time monitoring of water quality parameters across all sensor nodes
          </p>
        </div>

        {/* Sensor Selection */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          {sensors.map((sensor) => (
            <div
              key={sensor.id}
              onClick={() => setSelectedSensor(sensor)}
              className={`p-4 rounded-xl border-2 cursor-pointer transition-all ${
                selectedSensor?.id === sensor.id
                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                  : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <h3 className="font-semibold text-gray-800 dark:text-white">{sensor.name}</h3>
                <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                  sensor.status === 'active' ? 'bg-green-100 text-green-800' :
                  sensor.status === 'warning' ? 'bg-yellow-100 text-yellow-800' :
                  'bg-red-100 text-red-800'
                }`}>
                  {sensor.status}
                </span>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-300 mb-3">{sensor.location}</p>
              <div className="text-xs text-gray-500 dark:text-gray-400">
                Last update: {new Date(sensor.lastUpdate).toLocaleString()}
              </div>
            </div>
          ))}
        </div>

        {selectedSensor && (
          <>
            {/* Current Parameters */}
            <div className="bg-white dark:bg-secondary-dark-bg rounded-xl shadow-lg p-6 mb-8">
              <h3 className="text-lg font-semibold text-gray-800 dark:text-white mb-6">
                Current Parameters - {selectedSensor.name}
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {Object.entries(selectedSensor.parameters).map(([param, data]) => (
                  <div key={param} className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        {getParameterIcon(param)}
                        <span className="font-medium text-gray-800 dark:text-white capitalize">
                          {param.replace(/([A-Z])/g, ' $1').trim()}
                        </span>
                      </div>
                      {getStatusIcon(data.status)}
                    </div>
                    <div className="text-2xl font-bold text-gray-800 dark:text-white mb-1">
                      {data.value} {data.unit}
                    </div>
                    <div className="text-sm text-gray-600 dark:text-gray-300 mb-2">
                      Range: {data.min} - {data.max} {data.unit}
                    </div>
                    <div className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(data.status)}`}>
                      {data.status === 'good' ? 'Safe' : data.status === 'warning' ? 'Caution' : 'Unsafe'}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Historical Trends */}
            <div className="bg-white dark:bg-secondary-dark-bg rounded-xl shadow-lg p-6 mb-8">
              <h3 className="text-lg font-semibold text-gray-800 dark:text-white mb-6">
                Historical Trends - {selectedSensor.name}
              </h3>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* pH and Turbidity */}
                <div>
                  <h4 className="text-md font-medium text-gray-700 dark:text-gray-300 mb-4">pH & Turbidity</h4>
                  <ResponsiveContainer width="100%" height={250}>
                    <LineChart data={selectedSensor.historicalData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="time" />
                      <YAxis />
                      <Tooltip />
                      <Line type="monotone" dataKey="ph" stroke="#3b82f6" strokeWidth={2} name="pH" />
                      <Line type="monotone" dataKey="turbidity" stroke="#06b6d4" strokeWidth={2} name="Turbidity" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                {/* Bacterial Count and Chlorine */}
                <div>
                  <h4 className="text-md font-medium text-gray-700 dark:text-gray-300 mb-4">Bacterial Count & Chlorine</h4>
                  <ResponsiveContainer width="100%" height={250}>
                    <LineChart data={selectedSensor.historicalData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="time" />
                      <YAxis />
                      <Tooltip />
                      <Line type="monotone" dataKey="bacterialCount" stroke="#ef4444" strokeWidth={2} name="Bacterial Count" />
                      <Line type="monotone" dataKey="chlorineLevels" stroke="#10b981" strokeWidth={2} name="Chlorine Levels" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>

            
          </>
        )}
      </div>
    </div>
  );
};

export default WaterQualitySensors;
