import React, { useState } from 'react';
import { useLanguage } from '../contexts/LanguageContext';
import { useStateContext } from '../contexts/ContextProvider';
import { Button } from '../components';
import { FaExclamationCircle, FaLightbulb, FaFileAlt, FaMapMarkerAlt, FaPhone, FaEnvelope, FaCheckCircle } from 'react-icons/fa';

const ComplaintsAwareness = () => {
  const { t } = useLanguage();
  const { currentColor } = useStateContext();
  const [activeTab, setActiveTab] = useState('awareness');
  const [complaintForm, setComplaintForm] = useState({
    type: '',
    description: '',
    location: '',
    priority: 'medium',
    contactName: '',
    contactPhone: '',
    contactEmail: ''
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitStatus, setSubmitStatus] = useState(null);

  const complaintTypes = [
    { value: 'water_quality', label: 'Water Quality Issues' },
    { value: 'water_supply', label: 'Water Supply Problems' },
    { value: 'health_symptoms', label: 'Health Symptoms' },
    { value: 'infrastructure', label: 'Infrastructure Issues' },
    { value: 'billing', label: 'Billing Problems' },
    { value: 'other', label: 'Other' }
  ];

  const priorities = [
    { value: 'low', label: 'Low', color: 'green' },
    { value: 'medium', label: 'Medium', color: 'yellow' },
    { value: 'high', label: 'High', color: 'red' }
  ];

  const awarenessContent = [
    {
      id: 1,
      title: 'Safe Water Storage',
      category: 'Storage',
      content: 'Store water in clean, covered containers. Use clean utensils to draw water. Keep storage areas clean and dry.',
      image: '',
      tips: [
        'Use only food-grade containers for water storage',
        'Clean containers regularly with soap and water',
        'Keep containers covered to prevent contamination',
        'Use a clean ladle or tap to draw water'
      ]
    },
    {
      id: 2,
      title: 'Water Treatment Methods',
      category: 'Treatment',
      content: 'Learn effective methods to treat water at home for safe consumption.',
      image: '',
      tips: [
        'Boil water for at least 1 minute',
        'Use chlorine tablets as per instructions',
        'Solar disinfection (SODIS) method',
        'Use water filters certified for pathogen removal'
      ]
    },
    {
      id: 3,
      title: 'Hygiene Practices',
      category: 'Hygiene',
      content: 'Maintain proper hygiene to prevent waterborne diseases.',
      image: '',
      tips: [
        'Wash hands with soap before handling food',
        'Wash hands after using the toilet',
        'Keep kitchen and eating areas clean',
        'Use separate utensils for raw and cooked food'
      ]
    },
    {
      id: 4,
      title: 'Recognizing Contaminated Water',
      category: 'Detection',
      content: 'Learn to identify signs of contaminated water.',
      image: '⚠️',
      tips: [
        'Look for unusual color, taste, or odor',
        'Check for visible particles or cloudiness',
        'Be aware of unusual health symptoms in family',
        'Report suspicious water quality immediately'
      ]
    }
  ];

  const handleComplaintInputChange = (e) => {
    const { name, value } = e.target;
    setComplaintForm(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleComplaintSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      console.log('Complaint submitted:', complaintForm);
      
      setSubmitStatus('success');
      setComplaintForm({
        type: '',
        description: '',
        location: '',
        priority: 'medium',
        contactName: '',
        contactPhone: '',
        contactEmail: ''
      });
    } catch (error) {
      setSubmitStatus('error');
      console.error('Error submitting complaint:', error);
    } finally {
      setIsSubmitting(false);
      setTimeout(() => setSubmitStatus(null), 5000);
    }
  };

  return (
    <div className="mt-24 px-4 md:px-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-800 dark:text-white mb-4">
            <FaExclamationCircle className="inline-block mr-3" style={{ color: currentColor }} />
            Complaints & Awareness
          </h1>
          <p className="text-gray-600 dark:text-gray-300">
            Report issues and learn about safe water practices
          </p>
        </div>

        {/* Tab Navigation */}
        <div className="flex justify-center mb-8">
          <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-1">
         
            <button
              onClick={() => setActiveTab('awareness')}
              className={`px-6 py-3 rounded-md font-medium transition-all ${
                activeTab === 'awareness'
                  ? 'bg-white dark:bg-gray-700 text-gray-900 dark:text-white shadow-sm'
                  : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'
              }`}
            >
              <FaLightbulb className="inline-block mr-2" />
              Awareness Center
            </button>
          </div>
        </div>

        {/* Complaints Tab */}
       

        {/* Awareness Tab */}
        {activeTab === 'awareness' && (
          <div className="space-y-8">
            <div className="text-center mb-8">
              <h2 className="text-2xl font-semibold text-gray-800 dark:text-white mb-4">
                Water Safety & Hygiene Education
              </h2>
              <p className="text-gray-600 dark:text-gray-300">
                Learn essential practices for safe water handling and hygiene
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {awarenessContent.map((item) => (
                <div key={item.id} className="bg-white dark:bg-secondary-dark-bg rounded-xl shadow-lg p-6">
                  <div className="flex items-start space-x-4 mb-4">
                    <div className="text-4xl">{item.image}</div>
                    <div className="flex-1">
                      <h3 className="text-xl font-semibold text-gray-800 dark:text-white mb-2">
                        {item.title}
                      </h3>
                      <span className="inline-block px-3 py-1 bg-blue-100 dark:bg-blue-900/20 text-blue-800 dark:text-blue-300 text-sm font-medium rounded-full">
                        {item.category}
                      </span>
                    </div>
                  </div>
                  
                  <p className="text-gray-600 dark:text-gray-300 mb-4">
                    {item.content}
                  </p>
                  
                  <div className="space-y-2">
                    <h4 className="font-medium text-gray-800 dark:text-white">Key Tips:</h4>
                    <ul className="space-y-1">
                      {item.tips.map((tip, index) => (
                        <li key={index} className="flex items-start space-x-2 text-sm text-gray-600 dark:text-gray-300">
                          <FaCheckCircle className="text-green-500 mt-0.5 flex-shrink-0" />
                          <span>{tip}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              ))}
            </div>

            {/* Emergency Contacts */}
          <div className="bg-gradient-to-r from-red-500 to-red-700 p-6 rounded-2xl shadow-lg w-full max-w-md">
  <h2 className="text-white text-xl font-bold mb-3 flex items-center gap-2">
    🚨 Emergency Contact
  </h2>
  
  <div className="space-y-2 text-white">
    <p className="flex items-center gap-2">
      📞 <span className="font-medium">Emergency No:</span> 1800-180-5678
    </p>
    <p className="flex items-center gap-2">
      📧 <span className="font-medium">Email:</span> Ritesh_Kriplani@jeevanrakshak.com
    </p>
  </div>
</div>

          </div>
        )}
      </div>
    </div>
  );
};

export default ComplaintsAwareness;
