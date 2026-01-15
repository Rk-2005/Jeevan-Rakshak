import React, { useEffect, useState } from 'react';
import { ref, onValue, update } from 'firebase/database';
import { database, auth } from '../firebaseConfig';
import { signOut } from 'firebase/auth';
import { useNavigate } from 'react-router-dom';
import { Header } from '../components';
import './MyComplaints.css';

const MyComplaints = () => {
  const [complaints, setComplaints] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filter, setFilter] = useState('all');
  const [sortBy, setSortBy] = useState('date');
  const [userName, setUserName] = useState('');
  const [showFeedbackModal, setShowFeedbackModal] = useState(false);
  const [selectedComplaint, setSelectedComplaint] = useState(null);
  const [feedback, setFeedback] = useState({ rating: 5, comment: '' });
  const [feedbackSubmitting, setFeedbackSubmitting] = useState(false);
  const navigate = useNavigate();

  const userId = localStorage.getItem('userId');
  const userEmail = localStorage.getItem('userEmail');

  useEffect(() => {
    if (!userId) {
      navigate('/login');
      return;
    }

    // Get user name
    const userRef = ref(database, `users/${userId}`);
    onValue(userRef, (snapshot) => {
      if (snapshot.exists()) {
        setUserName(snapshot.val().name);
      }
    });

    // Get user's complaints
    const complaintsRef = ref(database, 'complaints');
    
    const unsubscribe = onValue(
      complaintsRef,
      (snapshot) => {
        try {
          const data = snapshot.val();
          if (data) {
            const complaintsArray = Object.entries(data)
              .map(([key, value]) => ({
                firebaseKey: key,
                ...value,
              }))
              .filter(complaint => complaint.userId === userId);
            setComplaints(complaintsArray);
          } else {
            setComplaints([]);
          }
          setLoading(false);
          setError(null);
        } catch (err) {
          console.error('Error processing complaints:', err);
          setError('Failed to load your complaints');
          setLoading(false);
        }
      },
      (err) => {
        console.error('Error fetching complaints:', err);
        setError('Failed to load complaints. Please refresh the page.');
        setLoading(false);
      }
    );

    return () => unsubscribe();
  }, [userId, navigate]);

  const handleLogout = async () => {
    try {
      await signOut(auth);
      localStorage.clear();
      navigate('/login');
    } catch (error) {
      console.error('Logout error:', error);
    }
  };

  const openFeedbackModal = (complaint) => {
    setSelectedComplaint(complaint);
    setFeedback({ 
      rating: complaint.feedback?.rating || 5, 
      comment: complaint.feedback?.comment || '' 
    });
    setShowFeedbackModal(true);
  };

  const closeFeedbackModal = () => {
    setShowFeedbackModal(false);
    setSelectedComplaint(null);
    setFeedback({ rating: 5, comment: '' });
  };

  const submitFeedback = async () => {
    if (!selectedComplaint) return;
    
    setFeedbackSubmitting(true);
    try {
      const complaintRef = ref(database, `complaints/${selectedComplaint.firebaseKey}`);
      await update(complaintRef, {
        feedback: {
          rating: feedback.rating,
          comment: feedback.comment,
          submittedAt: Date.now(),
          submittedDate: new Date().toISOString()
        }
      });
      
      alert('Thank you for your feedback!');
      closeFeedbackModal();
    } catch (error) {
      console.error('Error submitting feedback:', error);
      alert('Failed to submit feedback. Please try again.');
    } finally {
      setFeedbackSubmitting(false);
    }
  };

  const filteredComplaints = complaints
    .filter((complaint) => {
      if (filter === 'all') return true;
      return complaint.status?.toLowerCase() === filter;
    })
    .sort((a, b) => {
      if (sortBy === 'date') {
        return new Date(b.date) - new Date(a.date);
      }
      if (sortBy === 'status') {
        const statusOrder = { 'pending': 1, 'reviewed': 2, 'in-progress': 3, 'resolved': 4, 'closed': 5 };
        return (statusOrder[a.status?.toLowerCase()] || 0) - (statusOrder[b.status?.toLowerCase()] || 0);
      }
      return 0;
    });

  const getStatusClass = (status) => {
    switch (status?.toLowerCase()) {
      case 'pending':
        return 'status-pending';
      case 'reviewed':
        return 'status-reviewed';
      case 'in-progress':
      case 'in progress':
        return 'status-progress';
      case 'resolved':
        return 'status-resolved';
      case 'closed':
        return 'status-closed';
      default:
        return 'status-pending';
    }
  };

  const getPriorityClass = (priority) => {
    switch (priority?.toLowerCase()) {
      case 'high':
        return 'priority-high';
      case 'medium':
        return 'priority-medium';
      case 'low':
        return 'priority-low';
      default:
        return '';
    }
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  if (loading) {
    return (
      <div className="m-2 md:m-10 mt-24 p-2 md:p-10 bg-white rounded-3xl">
        <Header category="My Account" title="My Complaints" />
        <div className="loading-spinner">
          <div className="spinner"></div>
          <p>Loading your complaints...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="m-2 md:m-10 mt-24 p-2 md:p-10 bg-white rounded-3xl">
      <div className="user-header">
        <div>
          <Header category="My Account" title="My Complaints" />
          <p className="user-info">Welcome, {userName || userEmail}</p>
        </div>
        <div className="header-actions">
          <button onClick={() => navigate('/submit-complaint')} className="btn-primary">
            + New Complaint
          </button>
          <button onClick={handleLogout} className="btn-logout">
            Logout
          </button>
        </div>
      </div>

      {error && (
        <div className="alert alert-error">
          {error}
        </div>
      )}

      <div className="complaints-controls">
        <div className="filter-group">
          <label>Filter by Status:</label>
          <select value={filter} onChange={(e) => setFilter(e.target.value)}>
            <option value="all">All Status</option>
            <option value="pending">Pending</option>
            <option value="reviewed">Reviewed</option>
            <option value="in-progress">In Progress</option>
            <option value="resolved">Resolved</option>
            <option value="closed">Closed</option>
          </select>
        </div>

        <div className="filter-group">
          <label>Sort by:</label>
          <select value={sortBy} onChange={(e) => setSortBy(e.target.value)}>
            <option value="date">Date</option>
            <option value="status">Status</option>
          </select>
        </div>

        <div className="stats">
          <span className="stat-item">
            Total: <strong>{complaints.length}</strong>
          </span>
          <span className="stat-item pending">
            Pending: <strong>{complaints.filter((c) => c.status?.toLowerCase() === 'pending').length}</strong>
          </span>
          <span className="stat-item resolved">
            Resolved: <strong>{complaints.filter((c) => c.status?.toLowerCase() === 'resolved').length}</strong>
          </span>
        </div>
      </div>

      {filteredComplaints.length === 0 ? (
        <div className="no-complaints">
          <div className="empty-state">
            <svg width="100" height="100" viewBox="0 0 100 100" fill="none">
              <circle cx="50" cy="50" r="40" stroke="#e5e7eb" strokeWidth="3" />
              <path d="M50 30 L50 55 M35 50 L50 50" stroke="#9ca3af" strokeWidth="3" strokeLinecap="round" />
            </svg>
            <h3>No complaints found</h3>
            <p>You haven't submitted any complaints yet.</p>
            <button onClick={() => navigate('/submit-complaint')} className="btn-primary">
              Submit Your First Complaint
            </button>
          </div>
        </div>
      ) : (
        <div className="complaints-grid">
          {filteredComplaints.map((complaint) => (
            <div key={complaint.firebaseKey} className="complaint-card">
              <div className="card-header">
                <div className="complaint-id">#{complaint.id}</div>
                <div className="badges">
                  <span className={`priority-badge ${getPriorityClass(complaint.priority)}`}>
                    {complaint.priority || 'N/A'}
                  </span>
                  <span className={`status-badge ${getStatusClass(complaint.status)}`}>
                    {complaint.status || 'Pending'}
                  </span>
                </div>
              </div>

              <div className="card-body">
                <div className="complaint-type">{complaint.type}</div>
                
                <div className="complaint-info">
                  <div className="info-row">
                    <strong>Zone:</strong> {complaint.zone}
                  </div>
                  <div className="info-row">
                    <strong>Address:</strong> {complaint.address}
                  </div>
                  <div className="info-row">
                    <strong>Submitted:</strong> {formatDate(complaint.date)}
                  </div>
                  {complaint.userUrgency && (
                    <div className="info-row">
                      <strong>Your Urgency:</strong> {complaint.userUrgency}
                    </div>
                  )}
                </div>

                <div className="complaint-description">
                  <strong>Description:</strong>
                  <p>{complaint.description}</p>
                </div>

                {complaint.imageUrl && (
                  <div className="complaint-image">
                    <img src={complaint.imageUrl} alt="Complaint" />
                  </div>
                )}

                {complaint.adminNotes && (
                  <div className="admin-notes">
                    <strong>Admin Notes:</strong>
                    <p>{complaint.adminNotes}</p>
                  </div>
                )}

                {complaint.status === 'Closed' && (
                  <div className="feedback-section">
                    {complaint.feedback ? (
                      <div className="feedback-submitted">
                        <p className="feedback-label">✓ Feedback Submitted</p>
                        <div className="feedback-rating">Rating: {'⭐'.repeat(complaint.feedback.rating)}</div>
                      </div>
                    ) : (
                      <button 
                        onClick={() => openFeedbackModal(complaint)} 
                        className="btn-feedback"
                      >
                        📝 Give Feedback
                      </button>
                    )}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Feedback Modal */}
      {showFeedbackModal && selectedComplaint && (
        <div className="modal-overlay" onClick={closeFeedbackModal}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2>Submit Feedback</h2>
              <button className="close-btn" onClick={closeFeedbackModal}>×</button>
            </div>
            
            <div className="modal-body">
              <p className="feedback-complaint-id">Complaint ID: {selectedComplaint.id}</p>
              
              <div className="form-group">
                <label>Rating *</label>
                <div className="rating-selector">
                  {[1, 2, 3, 4, 5].map((star) => (
                    <button
                      key={star}
                      type="button"
                      className={`star-btn ${feedback.rating >= star ? 'active' : ''}`}
                      onClick={() => setFeedback({ ...feedback, rating: star })}
                    >
                      ⭐
                    </button>
                  ))}
                </div>
                <p className="rating-text">{feedback.rating} out of 5 stars</p>
              </div>

              <div className="form-group">
                <label>Comments</label>
                <textarea
                  value={feedback.comment}
                  onChange={(e) => setFeedback({ ...feedback, comment: e.target.value })}
                  placeholder="Share your experience with how your complaint was handled..."
                  rows="4"
                  className="feedback-textarea"
                />
              </div>
            </div>

            <div className="modal-footer">
              <button 
                onClick={closeFeedbackModal} 
                className="btn-secondary"
                disabled={feedbackSubmitting}
              >
                Cancel
              </button>
              <button 
                onClick={submitFeedback} 
                className="btn-primary"
                disabled={feedbackSubmitting}
              >
                {feedbackSubmitting ? 'Submitting...' : 'Submit Feedback'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default MyComplaints;
