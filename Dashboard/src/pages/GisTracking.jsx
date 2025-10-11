import { get, ref } from 'firebase/database';
import React, { useEffect, useState } from 'react';
import { Header } from '../components';
import { database } from '../firebaseConfig.js';

const GisTracking = () => {
  const [waterSavedValue, setWaterSavedValue] = useState('');
  const [variable2Value, setVariable2Value] = useState('');
  const [variable3Value, setVariable3Value] = useState('');
  const [variable4Value, setVariable4Value] = useState('');
  const [variable5Value, setVariable5Value] = useState('');

  useEffect(() => {
    const fetchValues = async () => {
      try {
        const waterSavedRef = ref(database, 'WaterSaved');
        const variable2Ref = ref(database, 'activeSensors');
        const variable3Ref = ref(database, 'frudsDetects');
        const variable4Ref = ref(database, 'leakdetects');
        const variable5Ref = ref(database, 'reportedComplains');

        const snapshot1 = await get(waterSavedRef);
        const snapshot2 = await get(variable2Ref);
        const snapshot3 = await get(variable3Ref);
        const snapshot4 = await get(variable4Ref);
        const snapshot5 = await get(variable5Ref);

        if (snapshot1.exists()) {
          setWaterSavedValue(snapshot1.val());
        }
        if (snapshot2.exists()) {
          setVariable2Value(snapshot2.val());
        }
        if (snapshot3.exists()) {
          setVariable3Value(snapshot3.val());
        }
        if (snapshot4.exists()) {
          setVariable4Value(snapshot4.val());
        }
        if (snapshot5.exists()) {
          setVariable5Value(snapshot5.val());
        }
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    };

    fetchValues();
  }, []);

  return (
    <div className="m-2 md:m-10 mt-24 p-2 md:p-10 bg-white rounded-3xl">
      <Header category="Page" title="Data" />
      <div>
        <p>Water Saved: {waterSavedValue}</p>
        <p>Variable 2: {variable2Value}</p>
        <p>Variable 3: {variable3Value}</p>
        <p>Variable 4: {variable4Value}</p>
        <p>Variable 5: {variable5Value}</p>
      </div>
    </div>
  );
};

export default GisTracking;
