import React, { useState } from 'react';
import { useLanguage } from '../contexts/LanguageContext';
import { useStateContext } from '../contexts/ContextProvider';
import { Button } from '../components';
import { FaUserMd, FaVirus, FaMapMarkerAlt, FaCalendarAlt } from 'react-icons/fa';
import { ref, push } from 'firebase/database';
import { database } from '../firebaseConfig';

const HealthDataCollection = () => {
  const { t } = useLanguage();
  const { currentColor } = useStateContext();
  const [formData, setFormData] = useState({
    villageName: '',
    patientAgeGroup: '',
    reportedSymptoms: [],
    confirmedDisease: '',
    reporterName: '',
    reporterContact: '',
    reportDate: new Date().toISOString().split('T')[0],
    additionalNotes: ''
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitStatus, setSubmitStatus] = useState(null);

  const ageGroups = [
    { value: '0-5', label: '0-5 years' },
    { value: '6-12', label: '6-12 years' },
    { value: '13-18', label: '13-18 years' },
    { value: '19-35', label: '19-35 years' },
    { value: '36-50', label: '36-50 years' },
    { value: '51-65', label: '51-65 years' },
    { value: '65+', label: '65+ years' }
  ];

  const symptoms = [
    { value: 'fever', label: 'Fever' },
    { value: 'diarrhea', label: 'Diarrhea' },
    { value: 'vomiting', label: 'Vomiting' },
    { value: 'nausea', label: 'Nausea' },
    { value: 'abdominal_pain', label: 'Abdominal Pain' },
    { value: 'headache', label: 'Headache' },
    { value: 'fatigue', label: 'Fatigue' },
    { value: 'muscle_aches', label: 'Muscle Aches' },
    { value: 'cough', label: 'Cough' },
    { value: 'sore_throat', label: 'Sore Throat' },
    { value: 'rash', label: 'Skin Rash' },
    { value: 'other', label: 'Other' }
  ];

  const diseases = [
    { value: '', label: 'Select Disease (Optional)' },
    { value: 'cholera', label: 'Cholera' },
    { value: 'typhoid', label: 'Typhoid' },
    { value: 'hepatitis_a', label: 'Hepatitis A' },
    { value: 'dysentery', label: 'Dysentery' },
    { value: 'gastroenteritis', label: 'Gastroenteritis' },
    { value: 'malaria', label: 'Malaria' },
    { value: 'dengue', label: 'Dengue' },
    { value: 'unknown', label: 'Unknown' }
  ];

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSymptomChange = (symptom) => {
    setFormData(prev => ({
      ...prev,
      reportedSymptoms: prev.reportedSymptoms.includes(symptom)
        ? prev.reportedSymptoms.filter(s => s !== symptom)
        : [...prev.reportedSymptoms, symptom]
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    
    try {
      const userId = localStorage.getItem('userId');
      const userEmail = localStorage.getItem('userEmail');
      
      // Validation
      if (!userId) {
        setSubmitStatus('error');
        setIsSubmitting(false);
        console.error('User not authenticated');
        return;
      }

      if (!formData.villageName.trim()) {
        setSubmitStatus('error');
        setIsSubmitting(false);
        console.error('Village name is required');
        return;
      }

      if (!formData.patientAgeGroup) {
        setSubmitStatus('error');
        setIsSubmitting(false);
        console.error('Patient age group is required');
        return;
      }

      if (formData.reportedSymptoms.length === 0) {
        setSubmitStatus('error');
        setIsSubmitting(false);
        console.error('At least one symptom must be selected');
        return;
      }

      if (!formData.reporterName.trim()) {
        setSubmitStatus('error');
        setIsSubmitting(false);
        console.error('Reporter name is required');
        return;
      }

      if (!formData.reporterContact.trim()) {
        setSubmitStatus('error');
        setIsSubmitting(false);
        console.error('Contact number is required');
        return;
      }

      // Prepare data for Firebase
      const healthReportData = {
        ...formData,
        userId,
        userEmail,
        submittedAt: new Date().toISOString(),
        submittedDate: new Date().toLocaleDateString(),
      };

      console.log('Submitting health data:', healthReportData);

      // Save to Firebase
      const healthReportsRef = ref(database, 'healthReports');
      const pushResult = await push(healthReportsRef, healthReportData);
      
      console.log('Health data submitted successfully with ID:', pushResult.key);
      
      setSubmitStatus('success');
      
      // Reset form
      setTimeout(() => {
        setFormData({
          villageName: '',
          patientAgeGroup: '',
          reportedSymptoms: [],
          confirmedDisease: '',
          reporterName: '',
          reporterContact: '',
          reportDate: new Date().toISOString().split('T')[0],
          additionalNotes: ''
        });
        setSubmitStatus(null);
      }, 2000);
    } catch (error) {
      setSubmitStatus('error');
      console.error('Error submitting health data:', error.message, error);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="mt-24 px-4 md:px-6">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-800 dark:text-white mb-4">
            <FaUserMd className="inline-block mr-3" style={{ color: currentColor }} />
            {t('healthDataCollection')}
          </h1>
          <p className="text-gray-600 dark:text-gray-300">
            Report health cases and symptoms for community monitoring
          </p>
        </div>

        {/* Form */}
        <div className="bg-white dark:bg-secondary-dark-bg rounded-xl shadow-lg p-6 md:p-8">
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Village Name */}
            <div>
              <label htmlFor="villageName" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                <FaMapMarkerAlt className="inline-block mr-2" />
                {t('villageName')} *
              </label>
              <input
                type="text"
                id="villageName"
                name="villageName"
                value={formData.villageName}
                onChange={handleInputChange}
                required
                className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:text-white"
                placeholder="Enter village name"
              />
            </div>

            {/* Patient Age Group */}
            <div>
              <label htmlFor="patientAgeGroup" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                <FaUserMd className="inline-block mr-2" />
                {t('patientAgeGroup')} *
              </label>
              <select
                id="patientAgeGroup"
                name="patientAgeGroup"
                value={formData.patientAgeGroup}
                onChange={handleInputChange}
                required
                className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:text-white"
              >
                <option value="">Select Age Group</option>
                {ageGroups.map(group => (
                  <option key={group.value} value={group.value}>
                    {group.label}
                  </option>
                ))}
              </select>
            </div>

            {/* Reported Symptoms */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                <FaVirus className="inline-block mr-2" />
                {t('reportedSymptoms')} *
              </label>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                {symptoms.map(symptom => (
                  <label key={symptom.value} className="flex items-center space-x-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={formData.reportedSymptoms.includes(symptom.value)}
                      onChange={() => handleSymptomChange(symptom.value)}
                      className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500"
                    />
                    <span className="text-sm text-gray-700 dark:text-gray-300">{symptom.label}</span>
                  </label>
                ))}
              </div>
            </div>

            {/* Confirmed Disease */}
            <div>
              <label htmlFor="confirmedDisease" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                {t('confirmedDisease')}
              </label>
              <select
                id="confirmedDisease"
                name="confirmedDisease"
                value={formData.confirmedDisease}
                onChange={handleInputChange}
                className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:text-white"
              >
                {diseases.map(disease => (
                  <option key={disease.value} value={disease.value}>
                    {disease.label}
                  </option>
                ))}
              </select>
            </div>

            {/* Reporter Information */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label htmlFor="reporterName" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Reporter Name *
                </label>
                <input
                  type="text"
                  id="reporterName"
                  name="reporterName"
                  value={formData.reporterName}
                  onChange={handleInputChange}
                  required
                  className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:text-white"
                  placeholder="Enter your name"
                />
              </div>
              <div>
                <label htmlFor="reporterContact" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Contact Number *
                </label>
                <input
                  type="tel"
                  id="reporterContact"
                  name="reporterContact"
                  value={formData.reporterContact}
                  onChange={handleInputChange}
                  required
                  className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:text-white"
                  placeholder="Enter contact number"
                />
              </div>
            </div>

            {/* Report Date */}
            <div>
              <label htmlFor="reportDate" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                <FaCalendarAlt className="inline-block mr-2" />
                Report Date *
              </label>
              <input
                type="date"
                id="reportDate"
                name="reportDate"
                value={formData.reportDate}
                onChange={handleInputChange}
                required
                className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:text-white"
              />
            </div>

            {/* Additional Notes */}
            <div>
              <label htmlFor="additionalNotes" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Additional Notes
              </label>
              <textarea
                id="additionalNotes"
                name="additionalNotes"
                value={formData.additionalNotes}
                onChange={handleInputChange}
                rows={4}
                className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:text-white"
                placeholder="Any additional information about the case..."
              />
            </div>

            {/* Submit Button */}
            <div className="flex justify-center">
              <Button
                type="submit"
                color="white"
                bgColor={currentColor}
                text={isSubmitting ? t('loading') : t('submitReport')}
                borderRadius="10px"
                size="lg"
                disabled={isSubmitting}
                className="px-8 py-3"
              />
            </div>

            {/* Status Messages */}
            {submitStatus === 'success' && (
              <div className="text-center p-4 bg-green-100 dark:bg-green-900/20 text-green-800 dark:text-green-300 rounded-lg animate-pulse">
                ✓ Health data submitted successfully!
              </div>
            )}
            {submitStatus === 'error' && (
              <div className="text-center p-4 bg-red-100 dark:bg-red-900/20 text-red-800 dark:text-red-300 rounded-lg border border-red-300 dark:border-red-700">
                ✗ Error submitting data. Please check all required fields and try again.
                <p className="text-xs mt-2 opacity-75">Check browser console for details.</p>
              </div>
            )}
          </form>
        </div>
      </div>
    </div>
  );
};

export default HealthDataCollection;
