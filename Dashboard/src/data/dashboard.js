import firebase from 'firebase/app';
import 'firebase/database';
import { BsBoxSeam, FiBarChart, HiOutlineRefresh, MdOutlineSupervisorAccount } from 'react-icons/all';

import firebaseConfig from '../firebaseConfig'; // Import your firebaseConfig file correctly

if (!firebase.apps.length) {
  firebase.initializeApp(firebaseConfig);
}

const db = firebase.database();
const dataRef = db.ref('Data');

const fetchEarningData = async () => {
  try {
    const snapshot = await dataRef.once('value');
    const firebaseData = snapshot.val();

    return [
      {
        icon: <MdOutlineSupervisorAccount />,
        amount: firebaseData.activeSensorsAmount || 'N/A',
        percentage: '-4%',
        title: 'Leak-Detects',
        iconColor: '#03C9D7',
        iconBg: '#E5FAFB',
        pcColor: 'red-600',
      },
      {
        icon: <BsBoxSeam />,
        amount: firebaseData.fraudDetectsAmount || 'N/A',
        percentage: '+23%',
        title: 'Fraud-Detects',
        iconColor: 'rgb(255, 244, 229)',
        iconBg: 'rgb(254, 201, 15)',
        pcColor: 'green-600',
      },
      {
        icon: <FiBarChart />,
        amount: firebaseData.reportedComplainsAmount || 'N/A',
        percentage: '+38%',
        title: 'Reported Complains',
        iconColor: 'rgb(228, 106, 118)',
        iconBg: 'rgb(255, 244, 229)',
        pcColor: 'green-600',
      },
      {
        icon: <HiOutlineRefresh />,
        amount: firebaseData.activeSensors || 'N/A',
        percentage: '-12%',
        title: 'Active Sensors',
        iconColor: 'rgb(0, 194, 146)',
        iconBg: 'rgb(235, 250, 242)',
        pcColor: 'red-600',
      },
    ];
  } catch (error) {
    console.error('Error fetching data from Firebase:', error);
    return []; // Return an empty array or handle error as needed
  }
};

export const earningData = fetchEarningData(); // Export the function or data correctly
