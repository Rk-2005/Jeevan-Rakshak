import React, { useState } from 'react';
import { signInWithEmailAndPassword, createUserWithEmailAndPassword } from 'firebase/auth';
import { ref, set } from 'firebase/database';
import { auth, database } from '../firebaseConfig';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Container,
  Typography,
  TextField,
  Button,
  Alert,
  CircularProgress,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Grid,
  Card,
  CardContent,
  Divider,
  IconButton,
  InputAdornment,
  Fade,
  Paper,
  alpha,
  styled,
  Tabs,
  Tab,
  Chip,
  Avatar,
  Stepper,
  Step,
  StepLabel,
  Stack
} from '@mui/material';
import {
  LockOutlined,
  PersonAdd,
  Visibility,
  VisibilityOff,
  Email,
  WaterDrop,
  HealthAndSafety,
  AdminPanelSettings,
  LocationOn,
  Person,
  Shield,
  ArrowForward,
  CheckCircle,
  ArrowBack,
  SmartToy,
  Analytics,
  Security,
  Dashboard
} from '@mui/icons-material';
import { Formik, Form, Field } from 'formik';
import * as Yup from 'yup';
import { motion, AnimatePresence } from 'framer-motion';
import './Auth.css';

// Styled Components
const GlassCard = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(4),
  borderRadius: 32,
  background: 'rgba(255, 255, 255, 0.98)',
  backdropFilter: 'blur(20px)',
  border: '1px solid rgba(255, 255, 255, 0.3)',
  boxShadow: `
    0 25px 50px -12px rgba(0, 102, 204, 0.1),
    0 8px 24px rgba(33, 150, 243, 0.1),
    inset 0 0 0 1px rgba(255, 255, 255, 0.5)
  `,
  position: 'relative',
  overflow: 'hidden',
  '&::before': {
    content: '""',
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    height: 6,
    background: 'linear-gradient(90deg, #0066cc 0%, #00bcd4 50%, #7b1fa2 100%)',
    borderRadius: '32px 32px 0 0'
  }
}));

const RoleCard = styled(Card)(({ theme, selected }) => ({
  padding: theme.spacing(3),
  borderRadius: 20,
  cursor: 'pointer',
  transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
  border: selected ? '2px solid #0066cc' : '2px solid transparent',
  backgroundColor: selected ? alpha('#0066cc', 0.05) : 'white',
  boxShadow: selected 
    ? '0 12px 32px rgba(0, 102, 204, 0.15)'
    : '0 4px 20px rgba(0, 0, 0, 0.05)',
  '&:hover': {
    transform: 'translateY(-4px)',
    boxShadow: '0 12px 32px rgba(0, 102, 204, 0.15)',
    border: '2px solid rgba(0, 102, 204, 0.3)'
  },
  '& .MuiAvatar-root': {
    width: 60,
    height: 60,
    margin: '0 auto 16px'
  }
}));

const AnimatedInput = styled(TextField)(({ theme }) => ({
  '& .MuiOutlinedInput-root': {
    borderRadius: 12,
    transition: 'all 0.3s ease',
    backgroundColor: 'rgba(255, 255, 255, 0.9)',
    '&:hover': {
      backgroundColor: 'white',
      '& .MuiOutlinedInput-notchedOutline': {
        borderColor: alpha('#0066cc', 0.5)
      }
    },
    '&.Mui-focused': {
      backgroundColor: 'white',
      '& .MuiOutlinedInput-notchedOutline': {
        borderColor: '#0066cc',
        borderWidth: 2
      }
    }
  }
}));

// Validation Schemas
const loginSchema = Yup.object().shape({
  email: Yup.string()
    .email('Please enter a valid email address')
    .required('Email is required'),
  password: Yup.string()
    .min(6, 'Password must be at least 6 characters')
    .required('Password is required')
});

const signupSchema = Yup.object().shape({
  name: Yup.string()
    .min(2, 'Name must be at least 2 characters')
    .required('Full name is required'),
  email: Yup.string()
    .email('Please enter a valid email address')
    .required('Email is required'),
  zone: Yup.string().required('Zone selection is required'),
  role: Yup.string().required('Please select a role'),
  password: Yup.string()
    .min(8, 'Password must be at least 8 characters')
    .matches(
      /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]/,
      'Must contain uppercase, lowercase, number and special character'
    )
    .required('Password is required'),
  confirmPassword: Yup.string()
    .oneOf([Yup.ref('password'), null], 'Passwords must match')
    .required('Please confirm your password')
});

const zones = [
  { value: 'zone1', label: 'Dharampeth', code: 'Zone 1' },
  { value: 'zone2', label: 'Sadar', code: 'Zone 2' },
  { value: 'zone3', label: 'Dhantoli', code: 'Zone 3' },
  { value: 'zone4', label: 'Hanuman Nagar', code: 'Zone 4' },
  { value: 'zone5', label: 'Gandhibagh', code: 'Zone 5' },
  { value: 'zone6', label: 'Laxmi Nagar', code: 'Zone 6' },
  { value: 'zone7', label: 'Ashi Nagar', code: 'Zone 7' },
  { value: 'zone8', label: 'Nehru Nagar', code: 'Zone 8' },
  { value: 'zone9', label: 'Lakadganj', code: 'Zone 9' },
  { value: 'zone10', label: 'Mangalwari', code: 'Zone 10' }
];

const roles = [
  {
    value: 'user',
    label: 'Citizen',
    icon: <Person />,
    description: 'Report issues, receive alerts',
    color: '#2196f3',
    features: ['Report Issues', 'View Updates', 'Get Alerts']
  },
  {
    value: 'admin',
    label: 'Administrator',
    icon: <AdminPanelSettings />,
    description: 'System management & monitoring',
    color: '#7b1fa2',
    features: ['Manage Zones', 'Analytics', 'System Control']
  },
  {
    value: 'asha-worker',
    label: 'ASHA Worker',
    icon: <HealthAndSafety />,
    description: 'Health reports & field data',
    color: '#00c853',
    features: ['Health Reports', 'Patient Data', 'Field Updates']
  }
];

const Login = () => {
  const [isLogin, setIsLogin] = useState(true);
  const [error, setError] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [activeTab, setActiveTab] = useState(0);
  const navigate = useNavigate();

  const handleLogin = async (values) => {
    try {
      setError('');
      const userCredential = await signInWithEmailAndPassword(auth, values.email, values.password);
      const user = userCredential.user;
      
      localStorage.setItem('userId', user.uid);
      localStorage.setItem('userEmail', user.email);
      
      const userRef = ref(database, `users/${user.uid}`);
      const { get } = await import('firebase/database');
      const snapshot = await get(userRef);
      
      if (snapshot.exists()) {
        const userData = snapshot.val();
        localStorage.setItem('userRole', userData.role || 'user');
        localStorage.setItem('userName', userData.name || 'User');
        localStorage.setItem('userZone', userData.zone || '');
        
        const role = userData.role || 'user';
        if (role === 'admin') {
          navigate('/dashboard');
        } else if (role === 'asha-worker') {
          navigate('/asha-dashboard');
        } else {
          navigate('/user-dashboard');
        }
      }
    } catch (error) {
      setError(error.code === 'auth/invalid-credential' 
        ? 'Invalid email or password. Please try again.' 
        : 'Unable to sign in. Please check your connection and try again.'
      );
    }
  };

  const handleSignup = async (values) => {
    try {
      setError('');
      const userCredential = await createUserWithEmailAndPassword(auth, values.email, values.password);
      const user = userCredential.user;

      await set(ref(database, `users/${user.uid}`), {
        name: values.name,
        email: values.email,
        role: values.role,
        zone: values.zone,
        createdAt: Date.now(),
      });

      localStorage.setItem('userId', user.uid);
      localStorage.setItem('userEmail', user.email);
      localStorage.setItem('userRole', values.role);
      localStorage.setItem('userName', values.name);
      localStorage.setItem('userZone', values.zone);

      if (values.role === 'admin') {
        navigate('/dashboard');
      } else if (values.role === 'asha-worker') {
        navigate('/asha-dashboard');
      } else {
        navigate('/user-dashboard');
      }
    } catch (error) {
      setError(error.code === 'auth/email-already-in-use'
        ? 'This email is already registered. Please sign in or use a different email.'
        : 'Unable to create account. Please try again.'
      );
    }
  };

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1
    }
  };

  const slideVariants = {
    enter: (direction) => ({
      x: direction > 0 ? 1000 : -1000,
      opacity: 0
    }),
    center: {
      zIndex: 1,
      x: 0,
      opacity: 1
    },
    exit: (direction) => ({
      zIndex: 0,
      x: direction < 0 ? 1000 : -1000,
      opacity: 0
    })
  };

  return (
    <Box sx={{ 
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #f5f9ff 0%, #e8f4ff 100%)',
      position: 'relative',
      overflow: 'hidden',
      py: 4
    }}>
      {/* Background elements */}
      <Box sx={{
        position: 'absolute',
        top: -100,
        right: -100,
        width: 400,
        height: 400,
        borderRadius: '50%',
        background: 'radial-gradient(circle, rgba(0, 102, 204, 0.1) 0%, transparent 70%)'
      }} />
      <Box sx={{
        position: 'absolute',
        bottom: -100,
        left: -100,
        width: 500,
        height: 500,
        borderRadius: '50%',
        background: 'radial-gradient(circle, rgba(123, 31, 162, 0.08) 0%, transparent 70%)'
      }} />

      <Container maxWidth="lg">
        <Grid container spacing={4} alignItems="center">
          {/* Left Side - Brand & Info */}
          <Grid item xs={12} md={5}>
            <motion.div
              initial={{ opacity: 0, x: -50 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6 }}
            >
              <Box sx={{ position: 'relative' }}>
                <Box sx={{ 
                  display: 'flex',
                  alignItems: 'center',
                  gap: 2,
                  mb: 4
                }}>
                  <Avatar sx={{ 
                    width: 64,
                    height: 64,
                    background: 'linear-gradient(135deg, #0066cc 0%, #00bcd4 100%)'
                  }}>
                    <WaterDrop sx={{ fontSize: 32 }} />
                  </Avatar>
                  <Box>
                    <Typography variant="h3" sx={{ 
                      fontWeight: 800,
                      background: 'linear-gradient(135deg, #0066cc 0%, #7b1fa2 100%)',
                      WebkitBackgroundClip: 'text',
                      WebkitTextFillColor: 'transparent',
                      mb: 1
                    }}>
                      Jeevan-Rakshak
                    </Typography>
                    <Typography variant="subtitle1" color="text.secondary" sx={{ fontWeight: 500 }}>
                      Smart Public Health & Water Monitoring System
                    </Typography>
                  </Box>
                </Box>

                <Typography variant="h5" sx={{ fontWeight: 600, mb: 3, color: '#1a237e' }}>
                  Protecting Nagpur's Health & Water Resources
                </Typography>

                <Stack spacing={2} sx={{ mb: 4 }}>
                  {[
                    { icon: '🌐', title: 'Real-time Monitoring', desc: 'Live tracking across 10 zones' },
                    { icon: '🚨', title: 'Smart Alerts', desc: 'Instant notifications for issues' },
                    { icon: '📊', title: 'AI Analytics', desc: 'Predictive insights & reports' },
                    { icon: '🤝', title: 'Community Driven', desc: 'Citizen participation platform' }
                  ].map((feature, index) => (
                    <motion.div
                      key={index}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.1 }}
                    >
                      <Box sx={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 2,
                        p: 2,
                        borderRadius: 3,
                        backgroundColor: 'rgba(255, 255, 255, 0.7)',
                        backdropFilter: 'blur(10px)',
                        border: '1px solid rgba(255, 255, 255, 0.3)',
                        boxShadow: '0 4px 12px rgba(0, 102, 204, 0.05)'
                      }}>
                        <Box sx={{ 
                          fontSize: 24,
                          width: 48,
                          height: 48,
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          borderRadius: 2,
                          backgroundColor: 'rgba(0, 102, 204, 0.1)'
                        }}>
                          {feature.icon}
                        </Box>
                        <Box>
                          <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 0.5 }}>
                            {feature.title}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            {feature.desc}
                          </Typography>
                        </Box>
                      </Box>
                    </motion.div>
                  ))}
                </Stack>

                <Box sx={{
                  background: 'linear-gradient(135deg, rgba(0, 102, 204, 0.1) 0%, rgba(123, 31, 162, 0.1) 100%)',
                  borderRadius: 3,
                  p: 3,
                  border: '1px solid rgba(0, 102, 204, 0.1)'
                }}>
                  <Grid container spacing={2} alignItems="center">
                    <Grid item xs={4}>
                      <Box sx={{ textAlign: 'center' }}>
                        <Typography variant="h4" sx={{ 
                          fontWeight: 800,
                          background: 'linear-gradient(135deg, #0066cc 0%, #7b1fa2 100%)',
                          WebkitBackgroundClip: 'text',
                          WebkitTextFillColor: 'transparent',
                          mb: 0.5
                        }}>
                          10
                        </Typography>
                        <Typography variant="caption" sx={{ fontWeight: 600, color: '#666' }}>
                          Zones
                        </Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={4}>
                      <Box sx={{ textAlign: 'center' }}>
                        <Typography variant="h4" sx={{ 
                          fontWeight: 800,
                          background: 'linear-gradient(135deg, #0066cc 0%, #7b1fa2 100%)',
                          WebkitBackgroundClip: 'text',
                          WebkitTextFillColor: 'transparent',
                          mb: 0.5
                        }}>
                          500+
                        </Typography>
                        <Typography variant="caption" sx={{ fontWeight: 600, color: '#666' }}>
                          Sensors
                        </Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={4}>
                      <Box sx={{ textAlign: 'center' }}>
                        <Typography variant="h4" sx={{ 
                          fontWeight: 800,
                          background: 'linear-gradient(135deg, #0066cc 0%, #7b1fa2 100%)',
                          WebkitBackgroundClip: 'text',
                          WebkitTextFillColor: 'transparent',
                          mb: 0.5
                        }}>
                          24/7
                        </Typography>
                        <Typography variant="caption" sx={{ fontWeight: 600, color: '#666' }}>
                          Monitoring
                        </Typography>
                      </Box>
                    </Grid>
                  </Grid>
                </Box>
              </Box>
            </motion.div>
          </Grid>

          {/* Right Side - Auth Form */}
          <Grid item xs={12} md={7}>
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5 }}
            >
              <GlassCard>
                <Box sx={{ textAlign: 'center', mb: 4 }}>
                  <Avatar sx={{ 
                    width: 80,
                    height: 80,
                    margin: '0 auto 16px',
                    background: 'linear-gradient(135deg, #0066cc 0%, #00bcd4 100%)',
                    boxShadow: '0 8px 24px rgba(0, 102, 204, 0.3)'
                  }}>
                    {isLogin ? (
                      <LockOutlined sx={{ fontSize: 40 }} />
                    ) : (
                      <PersonAdd sx={{ fontSize: 40 }} />
                    )}
                  </Avatar>
                  <Typography variant="h4" sx={{ 
                    fontWeight: 800,
                    background: 'linear-gradient(135deg, #0066cc 0%, #7b1fa2 100%)',
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent',
                    mb: 1
                  }}>
                    {isLogin ? 'Welcome Back' : 'Join Our Mission'}
                  </Typography>
                  <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
                    {isLogin 
                      ? 'Sign in to access your personalized dashboard'
                      : 'Create an account to contribute to community health'
                    }
                  </Typography>
                </Box>

                {/* Auth Mode Toggle */}
                <Box sx={{ 
                  display: 'flex', 
                  gap: 2, 
                  mb: 4,
                  p: 1,
                  borderRadius: 3,
                  backgroundColor: 'rgba(0, 102, 204, 0.05)'
                }}>
                  <Button
                    fullWidth
                    variant={isLogin ? 'contained' : 'text'}
                    onClick={() => setIsLogin(true)}
                    sx={{
                      py: 1.5,
                      borderRadius: 2,
                      fontWeight: 600,
                      textTransform: 'none',
                      fontSize: '1rem',
                      ...(isLogin && {
                        background: 'linear-gradient(135deg, #0066cc 0%, #00bcd4 100%)',
                        boxShadow: '0 4px 12px rgba(0, 102, 204, 0.3)'
                      })
                    }}
                  >
                    <LockOutlined sx={{ mr: 1 }} />
                    Sign In
                  </Button>
                  <Button
                    fullWidth
                    variant={!isLogin ? 'contained' : 'text'}
                    onClick={() => setIsLogin(false)}
                    sx={{
                      py: 1.5,
                      borderRadius: 2,
                      fontWeight: 600,
                      textTransform: 'none',
                      fontSize: '1rem',
                      ...(!isLogin && {
                        background: 'linear-gradient(135deg, #7b1fa2 0%, #e91e63 100%)',
                        boxShadow: '0 4px 12px rgba(123, 31, 162, 0.3)'
                      })
                    }}
                  >
                    <PersonAdd sx={{ mr: 1 }} />
                    Sign Up
                  </Button>
                </Box>

                <AnimatePresence mode="wait">
                  <motion.div
                    key={isLogin ? 'login' : 'signup'}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    transition={{ duration: 0.3 }}
                  >
                    {error && (
                      <Fade in={!!error}>
                        <Alert 
                          severity="error" 
                          sx={{ 
                            mb: 3, 
                            borderRadius: 2,
                            '& .MuiAlert-icon': {
                              alignItems: 'center'
                            }
                          }}
                          onClose={() => setError('')}
                        >
                          {error}
                        </Alert>
                      </Fade>
                    )}

                    <Formik
                      initialValues={{
                        email: '',
                        password: '',
                        confirmPassword: '',
                        name: '',
                        role: 'user',
                        zone: ''
                      }}
                      validationSchema={isLogin ? loginSchema : signupSchema}
                      onSubmit={isLogin ? handleLogin : handleSignup}
                    >
                      {({ values, errors, touched, handleChange, handleBlur, isSubmitting, isValid, setFieldValue }) => (
                        <Form>
                          <motion.div
                            variants={containerVariants}
                            initial="hidden"
                            animate="visible"
                          >
                            {!isLogin && (
                              <>
                                <motion.div variants={itemVariants}>
                                  <AnimatedInput
                                    fullWidth
                                    name="name"
                                    label="Full Name"
                                    value={values.name}
                                    onChange={handleChange}
                                    onBlur={handleBlur}
                                    error={touched.name && !!errors.name}
                                    helperText={touched.name && errors.name}
                                    sx={{ mb: 3 }}
                                    InputProps={{
                                      startAdornment: (
                                        <InputAdornment position="start">
                                          <Person color="action" />
                                        </InputAdornment>
                                      ),
                                    }}
                                  />
                                </motion.div>

                                <motion.div variants={itemVariants}>
                                  <AnimatedInput
                                    fullWidth
                                    name="email"
                                    label="Email Address"
                                    type="email"
                                    value={values.email}
                                    onChange={handleChange}
                                    onBlur={handleBlur}
                                    error={touched.email && !!errors.email}
                                    helperText={touched.email && errors.email}
                                    sx={{ mb: 3 }}
                                    placeholder="name@example.com"
                                    InputProps={{
                                      startAdornment: (
                                        <InputAdornment position="start">
                                          <Email color="action" />
                                        </InputAdornment>
                                      ),
                                    }}
                                  />
                                </motion.div>

                                <Grid container spacing={2} sx={{ mb: 3 }}>
                                  <Grid item xs={6}>
                                    <motion.div variants={itemVariants}>
                                      <AnimatedInput
                                        fullWidth
                                        name="password"
                                        label="Password"
                                        type={showPassword ? 'text' : 'password'}
                                        value={values.password}
                                        onChange={handleChange}
                                        onBlur={handleBlur}
                                        error={touched.password && !!errors.password}
                                        helperText={touched.password && errors.password}
                                        InputProps={{
                                          endAdornment: (
                                            <InputAdornment position="end">
                                              <IconButton
                                                onClick={() => setShowPassword(!showPassword)}
                                                edge="end"
                                                size="small"
                                              >
                                                {showPassword ? <VisibilityOff /> : <Visibility />}
                                              </IconButton>
                                            </InputAdornment>
                                          ),
                                        }}
                                      />
                                    </motion.div>
                                  </Grid>
                                  <Grid item xs={6}>
                                    <motion.div variants={itemVariants}>
                                      <AnimatedInput
                                        fullWidth
                                        name="confirmPassword"
                                        label="Confirm Password"
                                        type="password"
                                        value={values.confirmPassword}
                                        onChange={handleChange}
                                        onBlur={handleBlur}
                                        error={touched.confirmPassword && !!errors.confirmPassword}
                                        helperText={touched.confirmPassword && errors.confirmPassword}
                                      />
                                    </motion.div>
                                  </Grid>
                                </Grid>

                                <motion.div variants={itemVariants}>
                                  <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 600, color: '#1a237e' }}>
                                    Select Your Role
                                  </Typography>
                                  <Grid container spacing={2} sx={{ mb: 3 }}>
                                    {roles.map((role) => (
                                      <Grid item xs={4} key={role.value}>
                                        <RoleCard
                                          selected={values.role === role.value}
                                          onClick={() => setFieldValue('role', role.value)}
                                        >
                                          <Avatar sx={{ 
                                            background: role.color,
                                            mb: 2
                                          }}>
                                            {role.icon}
                                          </Avatar>
                                          <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 1 }}>
                                            {role.label}
                                          </Typography>
                                          <Typography variant="caption" color="text.secondary" sx={{ 
                                            display: 'block',
                                            mb: 1.5
                                          }}>
                                            {role.description}
                                          </Typography>
                                          <Box sx={{ 
                                            display: 'flex',
                                            flexDirection: 'column',
                                            gap: 0.5
                                          }}>
                                            {role.features.map((feature, idx) => (
                                              <Typography key={idx} variant="caption" sx={{ 
                                                color: 'text.secondary',
                                                display: 'flex',
                                                alignItems: 'center',
                                                gap: 0.5
                                              }}>
                                                <CheckCircle sx={{ fontSize: 12, color: role.color }} />
                                                {feature}
                                              </Typography>
                                            ))}
                                          </Box>
                                        </RoleCard>
                                      </Grid>
                                    ))}
                                  </Grid>
                                </motion.div>

                                <motion.div variants={itemVariants}>
                                  <FormControl fullWidth error={touched.zone && !!errors.zone} sx={{ mb: 3 }}>
                                    <InputLabel>Select Your Zone</InputLabel>
                                    <Select
                                      name="zone"
                                      value={values.zone}
                                      onChange={handleChange}
                                      onBlur={handleBlur}
                                      label="Select Your Zone"
                                      sx={{ borderRadius: 2 }}
                                      renderValue={(selected) => {
                                        const zone = zones.find(z => z.value === selected);
                                        return zone ? `${zone.code}: ${zone.label}` : '';
                                      }}
                                    >
                                      {zones.map((zone) => (
                                        <MenuItem key={zone.value} value={zone.value}>
                                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                                            <LocationOn sx={{ color: '#0066cc' }} />
                                            <Box>
                                              <Typography variant="body2" sx={{ fontWeight: 600 }}>
                                                {zone.label}
                                              </Typography>
                                              <Typography variant="caption" color="text.secondary">
                                                {zone.code}
                                              </Typography>
                                            </Box>
                                          </Box>
                                        </MenuItem>
                                      ))}
                                    </Select>
                                    {touched.zone && errors.zone && (
                                      <Typography color="error" variant="caption" sx={{ mt: 1, display: 'block' }}>
                                        {errors.zone}
                                      </Typography>
                                    )}
                                  </FormControl>
                                </motion.div>
                              </>
                            )}

                            {isLogin && (
                              <>
                                <motion.div variants={itemVariants}>
                                  <AnimatedInput
                                    fullWidth
                                    name="email"
                                    label="Email Address"
                                    type="email"
                                    value={values.email}
                                    onChange={handleChange}
                                    onBlur={handleBlur}
                                    error={touched.email && !!errors.email}
                                    helperText={touched.email && errors.email}
                                    sx={{ mb: 3 }}
                                    InputProps={{
                                      startAdornment: (
                                        <InputAdornment position="start">
                                          <Email color="action" />
                                        </InputAdornment>
                                      ),
                                    }}
                                  />
                                </motion.div>

                                <motion.div variants={itemVariants}>
                                  <AnimatedInput
                                    fullWidth
                                    name="password"
                                    label="Password"
                                    type={showPassword ? 'text' : 'password'}
                                    value={values.password}
                                    onChange={handleChange}
                                    onBlur={handleBlur}
                                    error={touched.password && !!errors.password}
                                    helperText={touched.password && errors.password}
                                    sx={{ mb: 1 }}
                                    InputProps={{
                                      endAdornment: (
                                        <InputAdornment position="end">
                                          <IconButton
                                            onClick={() => setShowPassword(!showPassword)}
                                            edge="end"
                                            size="small"
                                          >
                                            {showPassword ? <VisibilityOff /> : <Visibility />}
                                          </IconButton>
                                        </InputAdornment>
                                      ),
                                    }}
                                  />
                                  <Box sx={{ textAlign: 'right', mb: 3 }}>
                                    <Button
                                      variant="text"
                                      size="small"
                                      sx={{ textTransform: 'none', fontWeight: 500 }}
                                    >
                                      Forgot Password?
                                    </Button>
                                  </Box>
                                </motion.div>
                              </>
                            )}

                            <motion.div variants={itemVariants}>
                              <Button
                                type="submit"
                                fullWidth
                                variant="contained"
                                size="large"
                                disabled={isSubmitting || !isValid}
                                sx={{
                                  py: 2,
                                  borderRadius: 3,
                                  fontSize: '1rem',
                                  fontWeight: 700,
                                  textTransform: 'none',
                                  background: isLogin 
                                    ? 'linear-gradient(135deg, #0066cc 0%, #00bcd4 100%)'
                                    : 'linear-gradient(135deg, #7b1fa2 0%, #e91e63 100%)',
                                  boxShadow: '0 8px 24px rgba(0, 102, 204, 0.3)',
                                  '&:hover': {
                                    boxShadow: '0 12px 32px rgba(0, 102, 204, 0.4)',
                                    transform: 'translateY(-2px)'
                                  },
                                  '&.Mui-disabled': {
                                    background: '#e0e0e0'
                                  }
                                }}
                                endIcon={!isSubmitting && <ArrowForward />}
                              >
                                {isSubmitting ? (
                                  <CircularProgress size={24} color="inherit" />
                                ) : isLogin ? (
                                  'Sign In to Dashboard'
                                ) : (
                                  'Create Account & Continue'
                                )}
                              </Button>
                            </motion.div>
                          </motion.div>
                        </Form>
                      )}
                    </Formik>
                  </motion.div>
                </AnimatePresence>

                <Divider sx={{ my: 4, '&::before, &::after': { borderColor: 'rgba(0, 0, 0, 0.08)' } }}>
                  <Chip 
                    label="Nagpur Municipal Corporation" 
                    size="small"
                    sx={{ fontWeight: 500 }}
                  />
                </Divider>

                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="caption" color="text.secondary">
                    By continuing, you agree to our{' '}
                    <Button variant="text" size="small" sx={{ textTransform: 'none', fontWeight: 600 }}>
                      Terms of Service
                    </Button>{' '}
                    and{' '}
                    <Button variant="text" size="small" sx={{ textTransform: 'none', fontWeight: 600 }}>
                      Privacy Policy
                    </Button>
                  </Typography>
                </Box>
              </GlassCard>
            </motion.div>
          </Grid>
        </Grid>
      </Container>
    </Box>
  );
};

export default Login;
