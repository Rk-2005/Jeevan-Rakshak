import React, { useState } from 'react';

const SensorAllocation = () => {
  const [apiData, setApiData] = useState({ mapping: {}, sensor_nodes: [] });
  const [file, setFile] = useState(null);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleSubmit = (event) => {
    event.preventDefault();

    if (!file) {
      alert('Please select a file to upload.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    fetch('http://localhost:8080/sensor-allocation', {
      method: 'POST',
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        setApiData({
          mapping: data.mapping,
          sensor_nodes: data.sensor_nodes,
        });
      })
      .catch((error) => {
        console.error('Error fetching data:', error);
      });
  };

  return (
    <div className="m-2 md:m-10 mt-24 p-4 md:p-8 bg-white rounded-3xl shadow-lg">
      <form onSubmit={handleSubmit} style={{ marginBottom: '20px' }}>
        <input type="file" onChange={handleFileChange} style={{ marginBottom: '10px' }} />
        <button type="submit" style={{ padding: '5px 10px', borderRadius: '5px', border: 'none', backgroundColor: '#007bff', color: 'white', cursor: 'pointer' }}>Upload and Fetch Data</button>
      </form>
      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
        <div style={{ flex: 1, border: '1px solid rgba(0, 0, 0, 0.45)', padding: '10px', marginRight: '10px', borderRadius: '8px', boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)' }}>
          <h2 style={{ fontSize: '1.5rem', marginBottom: '10px', borderBottom: '1px solid rgba(0, 0, 0, 0.45)' }}>Mapping Data</h2>
          <ul style={{ listStyleType: 'none', fontSize: '1rem', lineHeight: '1.6' }}>
            {Object.entries(apiData.mapping).map(([key, value]) => (
              <li key={key} style={{ borderBottom: '1px solid rgba(0, 0, 0, 0.45)', padding: '5px 0' }}>
                {key} → {value}
              </li>
            ))}
          </ul>
        </div>
        <div style={{ flex: 1, border: '1px solid rgba(0, 0, 0, 0.45)', padding: '10px', marginLeft: '10px', borderRadius: '8px', boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)' }}>
          <h2 style={{ fontSize: '1.5rem', marginBottom: '10px', borderBottom: '1px solid rgba(0, 0, 0, 0.45)' }}>Sensor Nodes</h2>
          <ul style={{ listStyleType: 'none', fontSize: '1rem', lineHeight: '1.6' }}>
            {apiData.sensor_nodes.map((node, index) => (
              <li key={index} style={{ borderBottom: '1px solid rgba(0, 0, 0, 0.45)', padding: '5px 0' }}>
                Sensor Node: {node}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
};

export default SensorAllocation;
