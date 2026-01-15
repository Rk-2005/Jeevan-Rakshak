import React, { useState } from 'react';
import { ref, push, set } from 'firebase/database';
import { ref as storageRef, uploadBytes, getDownloadURL } from 'firebase/storage';
import { database, storage } from '../firebaseConfig';
import { Header } from '../components';
import './ComplaintForm.css';

const ComplaintForm = () => {
  const [formData, setFormData] = useState({
    name: '',
    phone: '',
    zone: '',
    address: '',
    complaintType: '',
    description: '',
    urgency: '',
  });
  const [image, setImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [submitStatus, setSubmitStatus] = useState({ show: false, type: '', message: '' });
  const [detectedPriority, setDetectedPriority] = useState('');

  const zones = [
    'Dharampeth (Zone 1)',
    'Laxmi Nagar (Zone 2)',
    'Hanuman Nagar (Zone 3)',
    'Dhantoli (Zone 4)',
    'Nehru Nagar (Zone 5)',
    'Gandhi Bagh (Zone 6)',
    'Sataranjipura (Zone 7)',
    'Lakadganj (Zone 8)',
    'Ashi Nagar (Zone 9)',
    'Mangalwari (Zone 10)',
  ];

  const complaintTypes = [
    'Water Quality Issue',
    'Health Alert',
    'Infrastructure Problem',
    'Water Leakage',
    'Contamination Report',
    'Supply Disruption',
    'Other',
  ];

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const analyzeComplaintPriority = async (complaintData) => {
    const GEMINI_API_KEY = 'AIzaSyC0w_7dClACJm6hlgBJW-gkIbezGvtm0uA';
    const GEMINI_API_URL = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${GEMINI_API_KEY}`;

    const prompt = `Analyze the following water/health complaint and determine its priority level (HIGH, MEDIUM, or LOW).

Complaint Details:
- Type: ${complaintData.complaintType}
- Zone: ${complaintData.zone}
- Address: ${complaintData.address}
- Description: ${complaintData.description}
- User Indicated Urgency: ${complaintData.urgency || 'Not specified'}

Priority Classification Guidelines:
- HIGH: Immediate health risks, disease outbreaks, severe contamination, multiple people affected, typhoid/cholera/dysentery cases
- MEDIUM: Water quality concerns, infrastructure issues affecting supply, potential health risks, localized problems
- LOW: Minor issues, cosmetic concerns, non-urgent maintenance requests

Respond with ONLY ONE WORD: HIGH, MEDIUM, or LOW`;

    try {
      const response = await fetch(GEMINI_API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          contents: [
            {
              parts: [
                {
                  text: prompt,
                },
              ],
            },
          ],
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to analyze priority');
      }

      const data = await response.json();
      const priorityText = data.candidates[0].content.parts[0].text.trim().toUpperCase();
      
      // Extract priority from response (HIGH, MEDIUM, or LOW)
      if (priorityText.includes('HIGH')) return 'High';
      if (priorityText.includes('MEDIUM')) return 'Medium';
      if (priorityText.includes('LOW')) return 'Low';
      
      // Default to Medium if unclear
      return 'Medium';
    } catch (error) {
      console.error('Error analyzing priority:', error);
      // Fallback to user's urgency or Medium
      if (complaintData.urgency === 'Critical') return 'High';
      if (complaintData.urgency === 'Low') return 'Low';
      return 'Medium';
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setSubmitStatus({ show: false, type: '', message: '' });

    try {
      // Check if user is logged in
      const userId = localStorage.getItem('userId');
      const userEmail = localStorage.getItem('userEmail');
      
      if (!userId) {
        throw new Error('Please login to submit a complaint');
      }

      // Validate form
      if (!formData.name || !formData.zone || !formData.address || !formData.complaintType || !formData.description) {
        throw new Error('Please fill in all required fields');
      }

      // Analyze priority using Gemini API
      const aiPriority = await analyzeComplaintPriority(formData);
      setDetectedPriority(aiPriority);

      let imageUrl = '';
      
      // Upload image if provided
      if (image) {
        const imageRef = storageRef(storage, `complaints/${Date.now()}_${image.name}`);
        await uploadBytes(imageRef, image);
        imageUrl = await getDownloadURL(imageRef);
      }

      // Prepare complaint data
      const complaintData = {
        id: `C${Date.now()}`,
        userId: userId,
        userEmail: userEmail,
        name: formData.name,
        phone: formData.phone,
        zone: formData.zone,
        address: formData.address,
        type: formData.complaintType,
        description: formData.description,
        userUrgency: formData.urgency,
        priority: aiPriority,
        imageUrl: imageUrl,
        status: 'Pending',
        date: new Date().toISOString(),
        createdAt: Date.now(),
      };

      // Save to Firebase Realtime Database
      const complaintsRef = ref(database, 'complaints');
      const newComplaintRef = push(complaintsRef);
      await set(newComplaintRef, complaintData);

      setSubmitStatus({
        show: true,
        type: 'success',
        message: `Complaint submitted successfully! Priority Level: ${aiPriority}`,
      });

      // Reset form
      setFormData({
        name: '',
        phone: '',
        zone: '',
        address: '',
        complaintType: '',
        description: '',
        urgency: '',
      });
      setImage(null);
      setImagePreview(null);
    } catch (error) {
      console.error('Error submitting complaint:', error);
      setSubmitStatus({
        show: true,
        type: 'error',
        message: error.message || 'Failed to submit complaint. Please try again.',
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="m-2 md:m-10 mt-24 p-2 md:p-10 bg-white rounded-3xl">
      <Header category="Form" title="Submit Complaint" />

      {submitStatus.show && (
        <div className={`alert ${submitStatus.type === 'success' ? 'alert-success' : 'alert-error'}`}>
          {submitStatus.message}
        </div>
      )}

      <form onSubmit={handleSubmit} className="complaint-form">
        <div className="form-grid">
          {/* Personal Information */}
          <div className="form-section">
            <h3 className="section-title">Personal Information</h3>
            
            <div className="form-group">
              <label htmlFor="name">Full Name *</label>
              <input
                type="text"
                id="name"
                name="name"
                value={formData.name}
                onChange={handleInputChange}
                placeholder="Enter your full name"
                required
              />
            </div>

       
            <div className="form-group">
              <label htmlFor="phone">Phone Number</label>
              <input
                type="tel"
                id="phone"
                name="phone"
                value={formData.phone}
                onChange={handleInputChange}
                placeholder="+91 XXXXX XXXXX"
              />
            </div>
          </div>

          {/* Complaint Details */}
          <div className="form-section">
            <h3 className="section-title">Complaint Details</h3>

            <div className="form-group">
              <label htmlFor="zone">Zone *</label>
              <select
                id="zone"
                name="zone"
                value={formData.zone}
                onChange={handleInputChange}
                required
              >
                <option value="">Select zone</option>
                {zones.map((zone) => (
                  <option key={zone} value={zone}>
                    {zone}
                  </option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label htmlFor="address">Address *</label>
              <input
                type="text"
                id="address"
                name="address"
                value={formData.address}
                onChange={handleInputChange}
                placeholder="Enter detailed address (street, building, etc.)"
                required
              />
            </div>

            <div className="form-group">
              <label htmlFor="complaintType">Complaint Type *</label>
              <select
                id="complaintType"
                name="complaintType"
                value={formData.complaintType}
                onChange={handleInputChange}
                required
              >
                <option value="">Select complaint type</option>
                {complaintTypes.map((type) => (
                  <option key={type} value={type}>
                    {type}
                  </option>
                ))}
              </select>
            </div>

          
          </div>
        </div>

        {/* Description */}
        <div className="form-section full-width">
          <div className="form-group">
            <label htmlFor="description">Complaint Description *</label>
            <textarea
              id="description"
              name="description"
              value={formData.description}
              onChange={handleInputChange}
              placeholder="Please provide detailed information about your complaint, including symptoms, affected people, timeline, etc."
              rows="6"
              required
            />
          </div>
        </div>

        {/* Image Upload */}
        <div className="form-section full-width">
          <div className="form-group">
            <label htmlFor="image">Upload Image (Optional)</label>
            <input
              type="file"
              id="image"
              accept="image/*"
              onChange={handleImageChange}
              className="file-input"
            />
            {imagePreview && (
              <div className="image-preview">
                <img src={imagePreview} alt="Preview" />
              </div>
            )}
          </div>
        </div>

        {/* Submit Button */}
        <div className="form-actions">
          <button
            type="submit"
            className="submit-btn"
            disabled={loading}
          >
            {loading ? (
              <>
                <span className="spinner"></span>
                Analyzing & Submitting...
              </>
            ) : (
              'Submit Complaint'
            )}
          </button>
        </div>

        {detectedPriority && (
          <div className="priority-badge">
            <span className={`badge badge-${detectedPriority.toLowerCase()}`}>
              AI Detected Priority: {detectedPriority}
            </span>
          </div>
        )}
      </form>
    </div>
  );
};

export default ComplaintForm;
