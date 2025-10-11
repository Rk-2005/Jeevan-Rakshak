// import { ColorPickerComponent } from '@syncfusion/ej2-react-inputs';
import { initializeApp } from 'firebase/app';
import { getStorage, ref, uploadBytes } from 'firebase/storage';
import React, { useState } from 'react';
import { Header } from '../components';
import { firebaseConfig } from '../firebaseConfig.js';

const firebaseApp = initializeApp(firebaseConfig);
const storage = getStorage(firebaseApp);

// const change = (args) => {
//   document.getElementById('preview').style.backgroundColor = args.currentValue.hex;
// };

// const CustomColorPicker = ({ id, mode }) => (
//   <ColorPickerComponent id={id} mode={mode} modeSwitcher={false} inline showButtons={false} change={change} />
// );

const ColorPicker = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadSuccess, setUploadSuccess] = useState(false);

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
  };

  const handleUpload = () => {
    if (selectedFile) {
      const storageRef = ref(storage, `files/${selectedFile.name}`);

      uploadBytes(storageRef, selectedFile).then(() => {
        setUploadSuccess(true); // Set the upload success state
      }).catch((error) => {
        console.error('Error uploading file:', error);
      });
    } else {
      console.error('No file selected');
    }
  };

  return (
    <div className="m-2 md:m-10 mt-24 p-2 md:p-10 bg-white rounded-3xl">
      <Header category="App" title="Color Picker" />
      <div className="text-center">
        <div id="preview" />
        <div className="flex justify-center items-center gap-20 flex-wrap">
          {/* Commented out the unwanted color pickers */}
          {/* <div>
            <p className="text-2xl font-semibold mt-2 mb-4">Inline Palette</p>
            <CustomColorPicker id="inline-palette" mode="Palette" />
          </div> */}
          <div>
            {/* <p className="text-2xl font-semibold mt-2 mb-4">Inline Picker</p> */}
            {/* <CustomColorPicker id="inline-picker" mode="Picker" /> */}
          </div>
        </div>
        <div>
          <input type="file" onChange={handleFileChange} />
          <button type="button" onClick={handleUpload}>Upload</button>
          {uploadSuccess && <p>File uploaded successfully!</p>}
        </div>
      </div>
    </div>
  );
};

export default ColorPicker;
