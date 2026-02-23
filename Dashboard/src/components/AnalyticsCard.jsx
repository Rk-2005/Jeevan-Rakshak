import React from 'react';

const AnalyticsCard = ({ title, value, unit, icon, status }) => {
  const isStatus = status !== undefined;

  return (
    <div className="bg-white dark:bg-secondary-dark-bg rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm p-5">
      <div className="flex items-center gap-3 mb-3">
        <span className="text-lg text-gray-500 dark:text-gray-400">{icon}</span>
        <p className="text-sm font-medium text-gray-500 dark:text-gray-400">{title}</p>
      </div>
      {isStatus ? (
        <span
          className={`inline-block px-3 py-1 rounded-md text-sm font-semibold ${
            status === 'Safe'
              ? 'bg-green-50 text-green-700 dark:bg-green-900/20 dark:text-green-400'
              : 'bg-red-50 text-red-700 dark:bg-red-900/20 dark:text-red-400'
          }`}
        >
          {status}
        </span>
      ) : (
        <p className="text-2xl font-bold text-gray-800 dark:text-white">
          {value}
          {unit && <span className="text-sm font-normal text-gray-400 ml-1">{unit}</span>}
        </p>
      )}
    </div>
  );
};

export default AnalyticsCard;
