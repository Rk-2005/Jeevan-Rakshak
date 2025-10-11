import { HtmlEditor, Image, Inject, Link, QuickToolbar, RichTextEditorComponent, Toolbar } from '@syncfusion/ej2-react-richtexteditor';
import { get, push, ref, set } from 'firebase/database';
import React, { useRef, useState } from 'react';
import { Header } from '../components';
import { EditorData } from '../data/dummy';
import { database } from '../firebaseConfig.js';

const Editor = () => {
  const [complain, setComplain] = useState('');
  const [name, setName] = useState('');
  const [imageBase64, setImageBase64] = useState(''); // State to hold base64 image

  // Ref to access the editor's instance
  const editorRef = useRef(null);

  const getCurrentReportedComplains = async (refObj) => {
    try {
      const snapshot = await get(refObj);
      return snapshot.exists() ? snapshot.val() : 0;
    } catch (error) {
      console.error('Error getting reportedComplains:', error);
      return 0;
    }
  };

  const updateReportedComplains = async (refObj, value) => {
    try {
      await set(refObj, value);
    } catch (error) {
      console.error('Error updating reportedComplains:', error);
    }
  };

  const handleAddData = async () => {
    try {
      const usersRef = ref(database, 'Complains');
      const rsRef = ref(database, 'reportedComplains');

      const newDataRef = push(usersRef);

      // Remove HTML tags using a regex before saving to Firebase
      const plainTextComplain = complain.replace(/<[^>]*>/g, ''); // Remove all HTML tags

      await set(newDataRef, {
        name,
        complain: plainTextComplain,
        imageBase64, // Save the image base64 string in Firebase
      });

      const currentReportedComplains = await getCurrentReportedComplains(rsRef);
      const incrementedValue = currentReportedComplains + 1;
      await updateReportedComplains(rsRef, incrementedValue);
      setName('');
      setComplain('');
      setImageBase64('');
      // Clear image after saving
      alert('Data Added Successfully');
    } catch (error) {
      console.error(error);
    }
  };

  // Function to update 'complain' state when the editor content changes
  const handleEditorChange = (args) => {
    setComplain(args.value);
  };

  // Function to handle image upload and convert to base64
  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    const reader = new FileReader();

    reader.onloadend = () => {
      // Read the file as a data URL and set it in state
      setImageBase64(reader.result);
    };

    if (file) {
      reader.readAsDataURL(file); // Convert image to base64
    }
  };

  return (
    <div className="m-2 md:m-10 mt-24 p-2 md:p-10 bg-white rounded-3xl">
      <Header category="App" title="Editor" />
      <div className="mb-4">
        <input
          type="text"
          placeholder="Name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          className="border-2 border-gray-300 p-2 rounded-lg focus:outline-none focus:ring-2 focus:ring-gray-200 focus:border-transparent"
        />
      </div>
      <div className="mb-4">
        <input
          type="file"
          accept="image/*"
          onChange={handleImageUpload}
          className="border-2 border-gray-300 p-2 rounded-lg focus:outline-none focus:ring-2 focus:ring-gray-200 focus:border-transparent"
        />
      </div>
      <RichTextEditorComponent
        ref={editorRef}
        value={complain}
        change={handleEditorChange}
      >
        <EditorData />
        <Inject services={[HtmlEditor, Toolbar, Image, Link, QuickToolbar]} />
      </RichTextEditorComponent>
      <div className="mb-4 mt-4">
        <button type="button" onClick={handleAddData} className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">
          Add Data
        </button>
      </div>
    </div>
  );
};

export default Editor;
