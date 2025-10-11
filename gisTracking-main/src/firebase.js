import { initializeApp } from 'firebase/app';
import { getFirestore } from 'firebase/firestore';
import { getStorage } from 'firebase/storage';
import firebase from "firebase/app";

const firebaseConfig = {
    apiKey: "AIzaSyB3qUPj3eEwLcPnOngYv7DCtVF0DZATJdQ",
    authDomain: "jalsanrakshan.firebaseapp.com",
    databaseURL: "https://jalsanrakshan-default-rtdb.asia-southeast1.firebasedatabase.app",
    projectId: "jalsanrakshan",
    storageBucket: "jalsanrakshan.appspot.com",
    messagingSenderId: "242091814600",
    appId: "1:242091814600:web:19e84d139121aea749fffd",
    measurementId: "G-R6G1YCWCMZ"
  };
  console.log("Firebase Config:", firebaseConfig);
const app = initializeApp(firebaseConfig);
console.log("Firebase Initialized"); // Log initialization success

const firestore = getFirestore(app);
const storage = getStorage(app);
console.log("Firestore Object:", firestore); 
export { firestore, storage};
