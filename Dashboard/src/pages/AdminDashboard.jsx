import React, { useEffect, useState } from 'react';
import { ref, onValue, update } from 'firebase/database';
import { database, auth } from '../firebaseConfig';
import { signOut } from 'firebase/auth';
import { useNavigate } from 'react-router-dom';
import { Header } from '../components';
import './AdminDashboard.css';

const AdminDashboard = () => {
  const [complaints, setComplaints] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filter, setFilter] = useState('all');
  const [sortBy, setSortBy] = useState('date');
  const [selectedComplaint, setSelectedComplaint] = useState(null);
  const [statusUpdate, setStatusUpdate] = useState('');
  const [adminNotes, setAdminNotes] = useState('');
  const [updating, setUpdating] = useState(false);
  const navigate = useNavigate();

  const userRole = localStorage.getItem('userRole');

  useEffect(() => {
    if (userRole !== 'admin') {
      navigate('/my-complaints');
      return;
    }

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
          setError('Failed to load complaints');
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
  }, [userRole, navigate]);

  const handleLogout = async () => {
    try {
      await signOut(auth);
      localStorage.clear();
      navigate('/login');
    } catch (error) {
      console.error('Logout error:', error);
    }
  };

  const openUpdateModal = (complaint) => {
    setSelectedComplaint(complaint);
    setStatusUpdate(complaint.status || 'Pending');
    setAdminNotes(complaint.adminNotes || '');
  };

  const closeUpdateModal = () => {
    setSelectedComplaint(null);
    setStatusUpdate('');
    setAdminNotes('');
  };

  const handleStatusUpdate = async () => {
    if (!selectedComplaint) return;

    setUpdating(true);
    try {
      const complaintRef = ref(database, `complaints/${selectedComplaint.firebaseKey}`);
      await update(complaintRef, {
        status: statusUpdate,
        adminNotes: adminNotes,
        lastUpdated: new Date().toISOString(),
        updatedBy: localStorage.getItem('userEmail'),
      });

      closeUpdateModal();
      alert('Complaint status updated successfully!');
    } catch (error) {
      console.error('Error updating complaint:', error);
      alert('Failed to update complaint status');
    } finally {
      setUpdating(false);
    }
  };

  const filteredComplaints = complaints
    .filter((complaint) => {
      if (filter === 'all') return true;
      if (filter === 'priority-high') return complaint.priority?.toLowerCase() === 'high';
      if (filter === 'priority-medium') return complaint.priority?.toLowerCase() === 'medium';
      if (filter === 'priority-low') return complaint.priority?.toLowerCase() === 'low';
      return complaint.status?.toLowerCase() === filter;
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
        <Header category="Admin" title="Dashboard" />
        <div className="loading-spinner">
          <div className="spinner"></div>
          <p>Loading complaints...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="m-2 md:m-10 mt-24 p-2 md:p-10 bg-white rounded-3xl">
      <div className="admin-header">
        <Header category="Admin" title="Complaints Management" />
        <button onClick={handleLogout} className="btn-logout">
          Logout
        </button>
      </div>

      {error && (
        <div className="alert alert-error">
          {error}
        </div>
      )}

      <div className="admin-stats">
        <div className="stat-card">
          <div className="stat-value">{complaints.length}</div>
          <div className="stat-label">Total Complaints</div>
        </div>
        <div className="stat-card pending">
          <div className="stat-value">{complaints.filter(c => c.status?.toLowerCase() === 'pending').length}</div>
          <div className="stat-label">Pending</div>
        </div>
        <div className="stat-card progress">
          <div className="stat-value">{complaints.filter(c => c.status?.toLowerCase() === 'in-progress').length}</div>
          <div className="stat-label">In Progress</div>
        </div>
        <div className="stat-card resolved">
          <div className="stat-value">{complaints.filter(c => c.status?.toLowerCase() === 'resolved').length}</div>
          <div className="stat-label">Resolved</div>
        </div>
        <div className="stat-card high">
          <div className="stat-value">{complaints.filter(c => c.priority?.toLowerCase() === 'high').length}</div>
          <div className="stat-label">High Priority</div>
        </div>
      </div>

      <div className="complaints-controls">
        <div className="filter-group">
          <label>Filter:</label>
          <select value={filter} onChange={(e) => setFilter(e.target.value)}>
            <option value="all">All Complaints</option>
            <optgroup label="By Status">
              <option value="pending">Pending</option>
              <option value="reviewed">Reviewed</option>
              <option value="in-progress">In Progress</option>
              <option value="resolved">Resolved</option>
              <option value="closed">Closed</option>
            </optgroup>
            <optgroup label="By Priority">
              <option value="priority-high">High Priority</option>
              <option value="priority-medium">Medium Priority</option>
              <option value="priority-low">Low Priority</option>
            </optgroup>
          </select>
        </div>

        <div className="filter-group">
          <label>Sort by:</label>
          <select value={sortBy} onChange={(e) => setSortBy(e.target.value)}>
            <option value="date">Date (Newest First)</option>
            <option value="priority">Priority (High to Low)</option>
          </select>
        </div>
      </div>

      <div className="complaints-grid">
        {filteredComplaints.map((complaint) => (
          <div key={complaint.firebaseKey} className="admin-complaint-card">
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
                  <strong>Submitted by:</strong> {complaint.name}
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
                <div className="admin-notes-display">
                  <strong>Admin Notes:</strong>
                  <p>{complaint.adminNotes}</p>
                </div>
              )}

              {complaint.lastUpdated && (
                <div className="last-updated">
                  Last updated: {formatDate(complaint.lastUpdated)}
                  {complaint.updatedBy && ` by ${complaint.updatedBy}`}
                </div>
              )}
            </div>

            <div className="card-actions">
              <button
                className="btn-update"
                onClick={() => openUpdateModal(complaint)}
              >
                Update Status
              </button>
            </div>
          </div>
        ))}
      </div>

      {/* Update Modal */}
      {selectedComplaint && (
        <div className="modal-overlay" onClick={closeUpdateModal}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Update Complaint Status</h3>
              <button className="modal-close" onClick={closeUpdateModal}>×</button>
            </div>

            <div className="modal-body">
              <div className="modal-info">
                <strong>Complaint ID:</strong> #{selectedComplaint.id}
              </div>
              <div className="modal-info">
                <strong>Type:</strong> {selectedComplaint.type}
              </div>
              <div className="modal-info">
                <strong>Submitted by:</strong> {selectedComplaint.name}
              </div>

              <div className="form-group">
                <label htmlFor="status">Update Status *</label>
                <select
                  id="status"
                  value={statusUpdate}
                  onChange={(e) => setStatusUpdate(e.target.value)}
                >
                  <option value="Pending">Pending</option>
                  <option value="Reviewed">Reviewed</option>
                  <option value="In Progress">In Progress</option>
                  <option value="Resolved">Resolved</option>
                  <option value="Closed">Closed</option>
                </select>
              </div>

              <div className="form-group">
                <label htmlFor="adminNotes">Admin Notes</label>
                <textarea
                  id="adminNotes"
                  value={adminNotes}
                  onChange={(e) => setAdminNotes(e.target.value)}
                  placeholder="Add notes for the user..."
                  rows="4"
                />
              </div>
            </div>

            <div className="modal-footer">
              <button className="btn-cancel" onClick={closeUpdateModal}>
                Cancel
              </button>
              <button
                className="btn-save"
                onClick={handleStatusUpdate}
                disabled={updating}
              >
                {updating ? 'Updating...' : 'Save Changes'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AdminDashboard;
