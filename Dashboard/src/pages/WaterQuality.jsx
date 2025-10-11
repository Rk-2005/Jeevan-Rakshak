  import React, { useState } from 'react';
  import { FaUpload, FaSpinner, FaCheckCircle, FaTimesCircle, FaInfoCircle, FaExclamationTriangle } from 'react-icons/fa';
  import { BsFillBarChartFill } from 'react-icons/bs';

  function WaterQuality() {
    const [file, setFile] = useState(null);
    const [predictions, setPredictions] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [fileName, setFileName] = useState('');
    const [showHelp, setShowHelp] = useState(false);
    const [showAlert, setShowAlert] = useState(false);

    const handleFileChange = (event) => {
      const selectedFile = event.target.files[0];
      if (selectedFile) {
        if (selectedFile.type === 'text/csv' || selectedFile.name.endsWith('.csv')) {
          setFile(selectedFile);
          setFileName(selectedFile.name);
          setError(null);
        } else {
          setError('Please upload a CSV file');
          setFile(null);
          setFileName('');
        }
      }
    };

    const handlePredict = async () => {
      if (!file) {
        setError('Please select a file first');
        return;
      }

      setLoading(true);
      setError(null);
      setPredictions([]);
      setShowAlert(false);
      
      try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('http://127.0.0.1:5200/predict', {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          throw new Error(`Server error: ${response.status}`);
        }

        const data = await response.json();
        
        if (!data.potability_prediction || !Array.isArray(data.potability_prediction)) {
          throw new Error('Invalid response format from server');
        }

        setPredictions(data.potability_prediction);
        
        // Show alert if any samples are non-potable
        if (data.potability_prediction.some(p => parseFloat(p) < 0.5)) {
          setShowAlert(true);
        }
      } catch (error) {
        console.error('Error:', error);
        setError(error.message || 'Failed to get predictions');
      } finally {
        setLoading(false);
      }
    };

    const getPotabilityStatus = (value) => {
      const numericValue = parseFloat(value);
      if (isNaN(numericValue)) return 'unknown';
      return numericValue >= 0.5 ? 'potable' : 'non-potable';
    };

    const potableCount = predictions.filter(p => parseFloat(p) >= 0.5).length;
    const nonPotableCount = predictions.length - potableCount;

    return (
      <div className="min-h-screen bg-gray-50 py-8 px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="max-w-4xl mx-auto bg-white text-white rounded-lg shadow-md p-6 mb-8">
          <div className="flex items-center justify-center gap-4">
            <h1 className="text-2xl font-bold">Water Quality Prediction</h1>
            <button 
              onClick={() => setShowHelp(!showHelp)}
              className="ml-auto text-white hover:text-blue-200"
              aria-label="Help"
            >
              <FaInfoCircle className="text-xl" />
            </button>
          </div>
        </div>

        {/* Help Modal */}
        {showHelp && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white rounded-lg shadow-xl max-w-md w-full p-6">
              <h3 className="text-lg font-bold text-blue-600 mb-4">How to use this tool</h3>
              <ol className="list-decimal pl-5 space-y-2 mb-4">
                <li>Prepare a CSV file with your water quality data</li>
                <li>Click "Choose CSV File" to select your file</li>
                <li>Click "Analyze Water Quality" to get predictions</li>
                <li>View results in the table below</li>
              </ol>
              <p className="text-sm text-gray-600">Note: Values ≥ 0.5 indicate potable water</p>
              <button 
                onClick={() => setShowHelp(false)}
                className="mt-4 bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded"
              >
                Close
              </button>
            </div>
          </div>
        )}

        {/* Alert Box for Unsafe Water */}
        {showAlert && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white rounded-lg shadow-xl max-w-md w-full p-6">
              <div className="flex items-start">
                <div className="flex-shrink-0 flex items-center justify-center h-12 w-12 rounded-full bg-red-100">
                  <FaExclamationTriangle className="h-6 w-6 text-red-600" />
                </div>
                <div className="ml-4">
                  <h3 className="text-lg font-medium text-gray-900">Water Quality Alert!</h3>
                  <div className="mt-2 text-sm text-gray-500">
                    <p>
                      {nonPotableCount} sample{nonPotableCount !== 1 ? 's' : ''} detected with unsafe water quality.
                      Please take appropriate action.
                    </p>
                  </div>
                  <div className="mt-4">
                    <button
                      type="button"
                      onClick={() => setShowAlert(false)}
                      className="inline-flex justify-center px-4 py-2 text-sm font-medium text-white bg-red-600 border border-transparent rounded-md hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500"
                    >
                      Acknowledge
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        <main className="max-w-4xl mx-auto space-y-8">
          {/* Upload Section */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <FaUpload /> Upload Water Quality Data
            </h2>
            
            <div className="flex flex-col sm:flex-row gap-4 items-start pt-3">
              <div className="flex-grow">
                <label className="block text-sm font-medium text-gray-700 mb-2 ml-1">
                  Select CSV File
                </label>
                <div className="flex items-center gap-2">
                  <input
                    type="file"
                    id="fileInput"
                    accept=".csv"
                    onChange={handleFileChange}
                    disabled={loading}
                    className="hidden"
                  />
                  <label
                    htmlFor="fileInput"
                    className={`flex-1 px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 ${
                      loading ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'
                    }`}
                  >
                    {fileName || 'Choose file...'}
                  </label>
                </div>
              </div>
              
              <button
                onClick={handlePredict}
                disabled={loading || !file}
                className={`mt-7 flex items-center gap-2 px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 ${
                  loading || !file ? 'opacity-50 cursor-not-allowed' : ''
                }`}
              >
                {loading ? (
                  <>
                    <FaSpinner className="animate-spin" /> Processing...
                  </>
                ) : (
                  <>
                    <BsFillBarChartFill /> Analyze Water Quality
                  </>
                )}
              </button>
            </div>

            {error && (
              <div className="mt-4 flex items-start gap-2 p-3 bg-red-50 border-l-4 border-red-500 text-red-700">
                <FaTimesCircle className="flex-shrink-0 mt-0.5" />
                <span>{error}</span>
              </div>
            )}
          </div>

          {/* Results Section */}
          {predictions.length > 0 && (
            <div className="bg-white rounded-lg shadow-md overflow-hidden">
              <div className="p-6 border-b border-gray-200">
                <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
                  <h2 className="text-xl font-semibold">Analysis Results</h2>
                  <div className="flex gap-4">
                    <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-green-100 text-green-800">
                      <FaCheckCircle className="mr-1" /> {potableCount} Safe
                    </span>
                    <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-red-100 text-red-800">
                      <FaTimesCircle className="mr-1" /> {nonPotableCount} Unsafe
                    </span>
                  </div>
                </div>
              </div>

              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Sample #
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Potability Score
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Status
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
  {predictions
    .map((value, index) => ({ index, value })) // Convert to an array of objects
    .sort((a, b) => parseFloat(a.value) - parseFloat(b.value)) // Sort: Non-potable first
    .map(({ value, index }) => {
      const status = getPotabilityStatus(value);
      return (
        <tr key={index} className={status === 'potable' ? 'bg-green-50' : 'bg-red-50'}>
          <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
            {index + 1}
          </td>
          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
            {parseFloat(value).toFixed(2)}
          </td>
          <td className="px-6 py-4 whitespace-nowrap">
            <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
              status === 'potable' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
            }`}>
              {status === 'potable' ? <FaCheckCircle className="mr-1" /> : <FaTimesCircle className="mr-1" />}
              {status === 'potable' ? 'Potable' : 'Non-Potable'}
            </span>
          </td>
        </tr>
      );
    })}
</tbody>

                </table>
              </div>

              <div className="p-4 bg-gray-50 border-t border-gray-200">
                <p className="text-sm text-gray-600">
                  <strong>Interpretation:</strong> Scores ≥ 0.5 indicate safe drinking water.
                </p>
              </div>
            </div>
          )}
        </main>
      </div>
    );
  }

  export default WaterQuality;