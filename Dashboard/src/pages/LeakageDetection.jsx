import React, { useState } from 'react';

// Minimal stub retained to satisfy historical imports; not used in routes.
// This version fixes ESLint issues and renders a note.
const LeakageDetection = () => {
  const [note] = useState('This module has been replaced by Outbreak Detection.');

  const items = [
    { sensor_node: 'A1', leak_detected: false },
    { sensor_node: 'B2', leak_detected: true },
  ];

  const sortedItems = items
    .slice()
    .sort((a, b) => {
      const aVal = a.leak_detected ? 1 : 0;
      const bVal = b.leak_detected ? 1 : 0;
      return bVal - aVal;
    });

  return (
    <div className="container mx-auto my-8 p-4 border rounded shadow">
      <h3 className="text-lg font-semibold mb-4">Leakage Detection (deprecated)</h3>
      <p className="text-sm text-gray-600 mb-4">{note}</p>
      <table className="w-full mx-auto">
        <thead>
          <tr>
            <th className="py-2 px-4 border-b">Sensor Node</th>
            <th className="py-2 px-4 border-b">Leak Detected</th>
          </tr>
        </thead>
        <tbody>
          {sortedItems.map((row, index) => {
            const isLeak = row.leak_detected;
            const rowClass = isLeak
              ? 'bg-red-500 text-white'
              : index % 2 === 0
                ? 'bg-white'
                : 'bg-gray-200';
            return (
              <tr key={row.sensor_node} className={rowClass}>
                <td className="py-2 px-4 border-b">{row.sensor_node}</td>
                <td className="py-2 px-4 border-b">{isLeak ? 'Yes' : 'No'}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
};

export default LeakageDetection;


