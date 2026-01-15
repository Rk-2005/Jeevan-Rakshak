import { initializeApp } from 'firebase/app';
import { getAnalytics } from 'firebase/analytics';
import { getAuth, GoogleAuthProvider } from 'firebase/auth';
import { getDatabase } from 'firebase/database';
import { getStorage } from 'firebase/storage';

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: 'AIzaSyA6SkgtZWE39ieQH1swYUnSXEdbFPrTf-c',
  authDomain: 'jalrakshak-dc076.firebaseapp.com',
  projectId: 'jalrakshak-dc076',
  storageBucket: 'jalrakshak-dc076.appspot.com',
  messagingSenderId: '454653778310',
  appId: '1:454653778310:web:2ec33bc7c096d723527407',
  measurementId: 'G-H6YVL455YC',
  databaseURL: 'https://jalrakshak-dc076-default-rtdb.firebaseio.com/',
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);
const auth = getAuth(app);
const database = getDatabase(app);
const storage = getStorage(app);
const googleProvider = new GoogleAuthProvider();

export { firebaseConfig, app, analytics, auth, database, storage, googleProvider };
export default firebaseConfig;
