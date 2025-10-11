import React, { useState } from 'react';

function App() {
  const [inputValue, setInputValue] = useState('');

  const handleInputChange = (e) => {
    setInputValue(e.target.value);
  };

  return (
    <div>
      <input
        type="text"
        value={inputValue}
        onChange={handleInputChange}
        placeholder="Type something..."
      />
      <input
        type="text"
        value={inputValue}
        readOnly
      />
      <input
        type="text"
        value={inputValue + inputValue}
        readOnly
      />
    </div>
  );
}

export default App;
