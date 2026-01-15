import React, { useEffect, useState } from 'react';
import { ref, onValue, remove } from 'firebase/database';
import { database } from '../firebaseConfig';
import { Header } from '../components';
import './SubmittedComplaints.css';

const SubmittedComplaints = () => {
  const [complaints, setComplaints] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('all');
  const [sortBy, setSortBy] = useState('date');
  const [error, setError] = useState(null);

  useEffect(() => {
    const complaintsRef = ref(database, 'complaints');
    
    const unsubscribe = onValue(
      complaintsRef,
      (snapshot) => {
        try {
          const data = snapshot.val();
          if (data) {
            const complaintsArray = Object.entries(data).map(([key, value]) => ({
              firebaseKey: key,
              ...value,
            }));
            setComplaints(complaintsArray);
          } else {
            setComplaints([]);
          }
          setLoading(false);
          setError(null);
        } catch (err) {
          console.error('Error processing complaints:', err);
          setError('Failed to process complaints data');
          setLoading(false);
        }
      },
      (err) => {
        console.error('Error fetching complaints:', err);
        setError('Failed to load complaints. Please check your internet connection and Firebase configuration.');
        setLoading(false);
      }
    );

    return () => unsubscribe();
  }, []);

  const handleDelete = async (firebaseKey) => {
    if (window.confirm('Are you sure you want to delete this complaint?')) {
      try {
        const complaintRef = ref(database, `complaints/${firebaseKey}`);
        await remove(complaintRef);
      } catch (error) {
        console.error('Error deleting complaint:', error);
        alert('Failed to delete complaint');
      }
    }
  };

  const filteredComplaints = complaints
    .filter((complaint) => {
      if (filter === 'all') return true;
      return complaint.priority?.toLowerCase() === filter;
    })
    .sort((a, b) => {
      if (sortBy === 'date') {
        return new Date(b.date) - new Date(a.date);
      }
      if (sortBy === 'priority') {
        const priorityOrder = { High: 3, Medium: 2, Low: 1 };
        return (priorityOrder[b.priority] || 0) - (priorityOrder[a.priority] || 0);
      }
      return 0;
    });

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
        <Header category="Complaints" title="All Submissions" />
        <div className="loading-spinner">
          <div className="spinner"></div>
          <p>Loading complaints...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="m-2 md:m-10 mt-24 p-2 md:p-10 bg-white rounded-3xl">
        <Header category="Complaints" title="All Submissions" />
        <div className="alert alert-error">
          {error}
        </div>
        <div className="no-complaints">
          <p>Please check:</p>
          <ul style={{ textAlign: 'left', maxWidth: '500px', margin: '1rem auto' }}>
            <li>Internet connection is active</li>
            <li>Firebase Realtime Database is enabled</li>
            <li>Database URL is correct in firebaseConfig.js</li>
            <li>Browser console for detailed errors</li>
          </ul>
          <button 
            onClick={() => window.location.reload()} 
            style={{ 
              padding: '0.75rem 1.5rem', 
              background: '#3b82f6', 
              color: 'white', 
              border: 'none', 
              borderRadius: '8px', 
              cursor: 'pointer',
              marginTop: '1rem'
            }}
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="m-2 md:m-10 mt-24 p-2 md:p-10 bg-white rounded-3xl">
      <Header category="Complaints" title="All Submissions" />

      <div className="complaints-controls">
        <div className="filter-group">
          <label>Filter by Priority:</label>
          <select value={filter} onChange={(e) => setFilter(e.target.value)}>
            <option value="all">All Priorities</option>
            <option value="high">High</option>
            <option value="medium">Medium</option>
            <option value="low">Low</option>
          </select>
        </div>

        <div className="filter-group">
          <label>Sort by:</label>
          <select value={sortBy} onChange={(e) => setSortBy(e.target.value)}>
            <option value="date">Date</option>
            <option value="priority">Priority</option>
          </select>
        </div>

        <div className="stats">
          <span className="stat-item">
            Total: <strong>{complaints.length}</strong>
          </span>
          <span className="stat-item high">
            High: <strong>{complaints.filter((c) => c.priority === 'High').length}</strong>
          </span>
          <span className="stat-item medium">
            Medium: <strong>{complaints.filter((c) => c.priority === 'Medium').length}</strong>
          </span>
          <span className="stat-item low">
            Low: <strong>{complaints.filter((c) => c.priority === 'Low').length}</strong>
          </span>
        </div>
      </div>

      {filteredComplaints.length === 0 ? (
        <div className="no-complaints">
          <p>No complaints found</p>
        </div>
      ) : (
        <div className="complaints-grid">
          {filteredComplaints.map((complaint) => (
            <div key={complaint.firebaseKey} className="complaint-card">
              <div className="card-header">
                <div className="complaint-id">#{complaint.id}</div>
                <span className={`priority-badge ${getPriorityClass(complaint.priority)}`}>
                  {complaint.priority || 'N/A'}
                </span>
              </div>

              <div className="card-body">
                <div className="complaint-type">{complaint.type}</div>
                
                <div className="complaint-info">
                  <div className="info-row">
                    <strong>Name:</strong> {complaint.name}
                  </div>
                  <div className="info-row">
                    <strong>Email:</strong> {complaint.email}
                  </div>
                  {complaint.phone && (
                    <div className="info-row">
                      <strong>Phone:</strong> {complaint.phone}
                    </div>
                  )}
                  <div className="info-row">
                    <strong>Zone:</strong> {complaint.zone}
                    <br />
                    <strong>Address:</strong> {complaint.address}
                  </div>
                  <div className="info-row">
                    <strong>Date:</strong> {formatDate(complaint.date)}
                  </div>
                  {complaint.userUrgency && (
                    <div className="info-row">
                      <strong>User Urgency:</strong> {complaint.userUrgency}
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

                <div className="complaint-status">
                  <span className={`status-badge status-${complaint.status?.toLowerCase() || 'pending'}`}>
                    {complaint.status || 'Pending'}
                  </span>
                </div>
              </div>

              <div className="card-actions">
                <button
                  className="delete-btn"
                  onClick={() => handleDelete(complaint.firebaseKey)}
                >
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default SubmittedComplaints;
