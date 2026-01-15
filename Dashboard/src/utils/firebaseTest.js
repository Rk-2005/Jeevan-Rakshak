// Firebase Connection Test
// Open browser console after running this to see results

import { database } from './firebaseConfig';
import { ref, set, get } from 'firebase/database';

export const testFirebaseConnection = async () => {
  console.log('🔍 Testing Firebase Connection...');
  
  try {
    // Test 1: Check if database instance exists
    console.log('✓ Database instance:', database ? 'OK' : 'FAILED');
    
    // Test 2: Try to write test data
    const testRef = ref(database, 'test_connection');
    await set(testRef, {
      timestamp: Date.now(),
      message: 'Connection test successful'
    });
    console.log('✓ Write test: OK');
    
    // Test 3: Try to read test data
    const snapshot = await get(testRef);
    if (snapshot.exists()) {
      console.log('✓ Read test: OK');
      console.log('Data:', snapshot.val());
    } else {
      console.log('⚠ Read test: No data found');
    }
    
    // Test 4: Check complaints path
    const complaintsRef = ref(database, 'complaints');
    const complaintsSnapshot = await get(complaintsRef);
    if (complaintsSnapshot.exists()) {
      const count = Object.keys(complaintsSnapshot.val()).length;
      console.log(`✓ Complaints path exists with ${count} entries`);
    } else {
      console.log('⚠ Complaints path is empty (this is OK if you haven\'t submitted any yet)');
    }
    
    console.log('✅ Firebase connection test completed successfully!');
    return true;
  } catch (error) {
    console.error('❌ Firebase connection test failed:', error);
    console.error('Error details:', {
      code: error.code,
      message: error.message,
      stack: error.stack
    });
    return false;
  }
};

// Auto-run test in development
if (process.env.NODE_ENV === 'development') {
  // Wait for DOM to load
  if (typeof window !== 'undefined') {
    window.testFirebaseConnection = testFirebaseConnection;
    console.log('💡 Run testFirebaseConnection() in console to test Firebase');
  }
}
