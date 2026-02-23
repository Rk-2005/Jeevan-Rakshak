import React, { useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import AnalyticsCard from '../components/AnalyticsCard';

const dummyWaterData = [
  { date: '2026-02-18', ph: 7.2, tds: 320, turbidity: 1.2 },
  { date: '2026-02-19', ph: 6.8, tds: 450, turbidity: 2.5 },
  { date: '2026-02-20', ph: 7.5, tds: 280, turbidity: 0.9 },
  { date: '2026-02-21', ph: 8.1, tds: 600, turbidity: 3.1 },
];

const computeAverages = (data) => {
  const len = data.length;
  if (len === 0) return { ph: 0, tds: 0, turbidity: 0 };
  const sum = data.reduce(
    (acc, d) => ({
      ph: acc.ph + d.ph,
      tds: acc.tds + d.tds,
      turbidity: acc.turbidity + d.turbidity,
    }),
    { ph: 0, tds: 0, turbidity: 0 }
  );
  return {
    ph: +(sum.ph / len).toFixed(2),
    tds: +(sum.tds / len).toFixed(0),
    turbidity: +(sum.turbidity / len).toFixed(2),
  };
};

const getWaterQualityStatus = (avg) => {
  const phSafe = avg.ph >= 6.5 && avg.ph <= 8.5;
  const tdsSafe = avg.tds < 500;
  const turbiditySafe = avg.turbidity < 2;
  return phSafe && tdsSafe && turbiditySafe ? 'Safe' : 'Unsafe';
};

const formatDate = (dateStr) => {
  const d = new Date(dateStr);
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
};

const HouseholdWaterAnalytics = () => {
  const waterData = dummyWaterData;

  const averages = useMemo(() => computeAverages(waterData), [waterData]);
  const qualityStatus = useMemo(() => getWaterQualityStatus(averages), [averages]);

  const chartData = useMemo(
    () => waterData.map((d) => ({ ...d, label: formatDate(d.date) })),
    [waterData]
  );

  return (
    <div className="mb-8">
      <h2 className="text-xl font-semibold text-gray-800 dark:text-white mb-4">
        Household Water Analytics
      </h2>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <AnalyticsCard title="Average pH Level" value={averages.ph} icon="⚗️" />
        <AnalyticsCard title="Average TDS Level" value={averages.tds} unit="ppm" icon="💧" />
        <AnalyticsCard title="Average Turbidity" value={averages.turbidity} unit="NTU" icon="🌊" />
        <AnalyticsCard title="Overall Water Quality" status={qualityStatus} icon="✅" />
      </div>

      <div className="bg-white dark:bg-secondary-dark-bg rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm p-5">
        <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-4">
          Water Quality Trends
        </h3>
        <div className="w-full" style={{ height: 300 }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis
                dataKey="label"
                tick={{ fontSize: 12, fill: '#6b7280' }}
                axisLine={{ stroke: '#d1d5db' }}
                tickLine={false}
              />
              <YAxis
                tick={{ fontSize: 12, fill: '#6b7280' }}
                axisLine={{ stroke: '#d1d5db' }}
                tickLine={false}
              />
              <Tooltip
                contentStyle={{
                  fontSize: 13,
                  borderRadius: 8,
                  border: '1px solid #e5e7eb',
                  boxShadow: '0 1px 3px rgba(0,0,0,0.08)',
                }}
              />
              <Legend
                iconType="circle"
                iconSize={8}
                wrapperStyle={{ fontSize: 13, paddingTop: 8 }}
              />
              <Line
                type="monotone"
                dataKey="ph"
                name="pH"
                stroke="#3b82f6"
                strokeWidth={2}
                dot={{ r: 3 }}
                activeDot={{ r: 5 }}
              />
              <Line
                type="monotone"
                dataKey="tds"
                name="TDS (ppm)"
                stroke="#f59e0b"
                strokeWidth={2}
                dot={{ r: 3 }}
                activeDot={{ r: 5 }}
              />
              <Line
                type="monotone"
                dataKey="turbidity"
                name="Turbidity (NTU)"
                stroke="#10b981"
                strokeWidth={2}
                dot={{ r: 3 }}
                activeDot={{ r: 5 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

export default HouseholdWaterAnalytics;
