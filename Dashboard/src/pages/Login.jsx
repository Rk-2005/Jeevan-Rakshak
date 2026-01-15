import React, { useState } from 'react';
import { signInWithEmailAndPassword, createUserWithEmailAndPassword } from 'firebase/auth';
import { ref, set } from 'firebase/database';
import { auth, database } from '../firebaseConfig';
import { useNavigate } from 'react-router-dom';
import './Auth.css';

const Login = () => {
  const [isLogin, setIsLogin] = useState(true);
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    confirmPassword: '',
    name: '',
    role: 'user',
    zone: '',
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleLogin = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      const userCredential = await signInWithEmailAndPassword(auth, formData.email, formData.password);
      const user = userCredential.user;
      
      // Store user info in localStorage
      localStorage.setItem('userId', user.uid);
      localStorage.setItem('userEmail', user.email);
      
      // Redirect based on role (check from database)
      const userRef = ref(database, `users/${user.uid}`);
      const { get } = await import('firebase/database');
      const snapshot = await get(userRef);
      
      if (snapshot.exists()) {
        const userData = snapshot.val();
        localStorage.setItem('userRole', userData.role || 'user');
        localStorage.setItem('userName', userData.name || 'User');
        localStorage.setItem('userZone', userData.zone || '');
        
        if (userData.role === 'admin') {
          navigate('/dashboard');
        } else if (userData.role === 'asha-worker') {
          navigate('/asha-dashboard');
        } else {
          navigate('/user-dashboard');
        }
      } else {
        navigate('/user-dashboard');
      }
    } catch (error) {
      console.error('Login error:', error);
      setError(error.message || 'Failed to login. Please check your credentials.');
    } finally {
      setLoading(false);
    }
  };

  const handleSignup = async (e) => {
    e.preventDefault();
    setError('');

    if (formData.password !== formData.confirmPassword) {
      setError('Passwords do not match');
      return;
    }

    if (formData.password.length < 6) {
      setError('Password must be at least 6 characters');
      return;
    }

    setLoading(true);

    try {
      const userCredential = await createUserWithEmailAndPassword(auth, formData.email, formData.password);
      const user = userCredential.user;

      // Save user data to database
      await set(ref(database, `users/${user.uid}`), {
        name: formData.name,
        email: formData.email,
        role: formData.role,
        zone: formData.zone,
        createdAt: Date.now(),
      });

      // Store user info in localStorage
      localStorage.setItem('userId', user.uid);
      localStorage.setItem('userEmail', user.email);
      localStorage.setItem('userRole', formData.role);
      localStorage.setItem('userName', formData.name);
      localStorage.setItem('userZone', formData.zone);

      // Redirect based on role
      if (formData.role === 'admin') {
        navigate('/dashboard');
      } else if (formData.role === 'asha-worker') {
        navigate('/asha-dashboard');
      } else {
        navigate('/user-dashboard');
      }
    } catch (error) {
      console.error('Signup error:', error);
      setError(error.message || 'Failed to create account. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-container">
      <div className="auth-card">
        <div className="auth-header">
          <h2>{isLogin ? 'Login' : 'Create Account'}</h2>
          <p>{isLogin ? 'Welcome back! Please login to continue.' : 'Join us to submit and track complaints'}</p>
        </div>

        {error && (
          <div className="auth-error">
            {error}
          </div>
        )}

        <form onSubmit={isLogin ? handleLogin : handleSignup} className="auth-form">
          {!isLogin && (
            <>
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
                <label htmlFor="zone">Zone *</label>
                <select
                  id="zone"
                  name="zone"
                  value={formData.zone}
                  onChange={handleInputChange}
                  required
                >
                  <option value="">Select Zone</option>
                  <option value="Dharampeth (Zone 1)">Dharampeth (Zone 1)</option>
                  <option value="Sadar (Zone 2)">Sadar (Zone 2)</option>
                  <option value="Dhantoli (Zone 3)">Dhantoli (Zone 3)</option>
                  <option value="Hanuman Nagar (Zone 4)">Hanuman Nagar (Zone 4)</option>
                  <option value="Gandhibagh (Zone 5)">Gandhibagh (Zone 5)</option>
                  <option value="Laxmi Nagar (Zone 6)">Laxmi Nagar (Zone 6)</option>
                  <option value="Ashi Nagar (Zone 7)">Ashi Nagar (Zone 7)</option>
                  <option value="Nehru Nagar (Zone 8)">Nehru Nagar (Zone 8)</option>
                  <option value="Lakadganj (Zone 9)">Lakadganj (Zone 9)</option>
                  <option value="Mangalwari (Zone 10)">Mangalwari (Zone 10)</option>
                </select>
              </div>
            </>
          )}

          <div className="form-group">
            <label htmlFor="email">Email *</label>
            <input
              type="email"
              id="email"
              name="email"
              value={formData.email}
              onChange={handleInputChange}
              placeholder="your.email@example.com"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="password">Password *</label>
            <input
              type="password"
              id="password"
              name="password"
              value={formData.password}
              onChange={handleInputChange}
              placeholder="Enter password (min 6 characters)"
              required
            />
          </div>

          {!isLogin && (
            <>
              <div className="form-group">
                <label htmlFor="confirmPassword">Confirm Password *</label>
                <input
                  type="password"
                  id="confirmPassword"
                  name="confirmPassword"
                  value={formData.confirmPassword}
                  onChange={handleInputChange}
                  placeholder="Re-enter password"
                  required
                />
              </div>

              <div className="form-group">
                <label htmlFor="role">Account Type *</label>
                <select
                  id="role"
                  name="role"
                  value={formData.role}
                  onChange={handleInputChange}
                  required
                >
                  <option value="user">User</option>
                  <option value="admin">Admin</option>
                  <option value="asha-worker">ASHA Worker</option>
                </select>
              </div>
            </>
          )}

          <button
            type="submit"
            className="auth-submit-btn"
            disabled={loading}
          >
            {loading ? (
              <>
                <span className="spinner"></span>
                {isLogin ? 'Logging in...' : 'Creating account...'}
              </>
            ) : (
              isLogin ? 'Login' : 'Sign Up'
            )}
          </button>
        </form>

        <div className="auth-toggle">
          <p>
            {isLogin ? "Don't have an account? " : "Already have an account? "}
            <button onClick={() => {
              setIsLogin(!isLogin);
              setError('');
              setFormData({
                email: '',
                password: '',
                confirmPassword: '',
                name: '',
                role: 'user',
                zone: '',
              });
            }}>
              {isLogin ? 'Sign Up' : 'Login'}
            </button>
          </p>
        </div>
      </div>
    </div>
  );
};

export default Login;
