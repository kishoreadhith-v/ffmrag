import React, { useState, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import './App.css';

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [results, setResults] = useState(null);
  const [description, setDescription] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingDescription, setLoadingDescription] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);
  const [chatMessage, setChatMessage] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [chatLoading, setChatLoading] = useState(false);
  const chatContainerRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      setPreview(URL.createObjectURL(file));
      setResults(null);
      setDescription(null);
      setError(null);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      setImage(file);
      setPreview(URL.createObjectURL(file));
      setResults(null);
      setDescription(null);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!image) return;
    setLoading(true);
    setLoadingDescription(true);
    setError(null);
    setResults(null);
    setDescription(null);
    
    try {
      const formData = new FormData();
      formData.append('image', image);
      const response = await fetch('http://localhost:5000/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to analyze image');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const lines = decoder.decode(value).split('\n').filter(line => line.trim());
        for (const line of lines) {
          try {
            const data = JSON.parse(line);
            
            if (data.type === 'results') {
              setResults(data.data);
              setLoading(false);  // Stop loading spinner for results
            } 
            else if (data.type === 'description') {
              setDescription(data.data);
              setLoadingDescription(false);  // Stop loading spinner for description
            }
            else if (data.type === 'error') {
              throw new Error(data.data);
            }
          } catch (e) {
            console.error('Error parsing response:', e);
          }
        }
      }
    } catch (err) {
      setError(err.message);
      setLoading(false);
      setLoadingDescription(false);
    }
  };

  const handleChatSubmit = async (e) => {
    e.preventDefault();
    if (!chatMessage.trim()) return;

    const message = chatMessage.trim();
    setChatMessage('');
    setChatLoading(true);

    // Add user message to chat
    setChatHistory(prev => [...prev, { type: 'user', message }]);

    try {
      const response = await fetch('http://localhost:5500/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message }),
      });

      if (!response.ok) {
        throw new Error('Failed to get response');
      }

      const data = await response.json();
      
      if (data.status === 'success') {
        // Add bot response to chat
        setChatHistory(prev => [...prev, { 
          type: 'bot', 
          message: data.response 
        }]);
      } else {
        throw new Error(data.error || 'Failed to get response');
      }
    } catch (err) {
      setChatHistory(prev => [...prev, { 
        type: 'error', 
        message: err.message || 'Sorry, I encountered an error. Please try again.' 
      }]);
    } finally {
      setChatLoading(false);
    }
  };

  const handleSearchSpecies = () => {
    if (!results || !results[0]) return;
    const speciesName = results[0].species;
    setChatMessage(`Tell me about ${speciesName}`);
    handleChatSubmit({ preventDefault: () => {} });
  };

  // Scroll to bottom of chat when new messages arrive
  React.useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [chatHistory]);

  return (
    <div className="container">
      <div className="card">
        <h1 className="title">🔬 Species Analyzer AI</h1>
        <p className="subtitle">
          Upload any image to identify species using advanced machine learning
        </p>
        
        <div 
          className="upload-section"
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
          onClick={() => fileInputRef.current?.click()}
        >
          {!preview ? (
            <>
              <div className="upload-icon">📸</div>
              <p className="upload-text">
                Drag & drop your image here or click to browse
              </p>
            </>
          ) : (
            <img src={preview} alt="Preview" className="preview-image" />
          )}
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="file-input"
          />
        </div>

        <button 
          className="upload-button"
          onClick={handleUpload} 
          disabled={!image || loading}
        >
          {loading ? (
            <>
              <span className="loading-spinner"></span>
              Analyzing Image...
            </>
          ) : 'Analyze Image'}
        </button>

        {error && (
          <div className="error-message">
            ❌ {error}
          </div>
        )}

        {results && (
          <div className="results-section">
            <h2 className="results-title">Analysis Results</h2>
            
            {/* Top result with description */}
            {results[0] && (
              <div className="top-result">
                <div className="result-item">
                  <div className="species-name">
                    🏆 {results[0].species}
                    <span className="confidence-value">
                      ({(results[0].confidence).toFixed(1)}%)
                    </span>
                  </div>
                  <div className="group-tag">
                    {results[0].group}
                  </div>
                  <div className="note-text">
                    {results[0].note}
                  </div>
                  <button 
                    className="search-button"
                    onClick={handleSearchSpecies}
                    disabled={chatLoading}
                  >
                    🔍 Search for Species Info
                  </button>
                </div>
                
                <div className="species-description">
                  <h3>About this Species</h3>
                  {loadingDescription ? (
                    <div className="description-loading">
                      <span className="loading-spinner"></span>
                      Loading detailed description...
                    </div>
                  ) : description ? (
                    <ReactMarkdown>{description}</ReactMarkdown>
                  ) : null}
                </div>
              </div>
            )}
            
            {/* Other results */}
            <div className="other-results">
              <h3>Other Possibilities</h3>
              {results.slice(1).map((res, idx) => (
                <div key={res.class_id} className="result-item">
                  <div className="species-name">
                    🔍 {res.species}
                    <span className="confidence-value">
                      ({(res.confidence).toFixed(1)}%)
                    </span>
                  </div>
                  <div className="group-tag">
                    {res.group}
                  </div>
                  <div className="note-text">
                    {res.note}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Chat Interface */}
        <div className="chat-section">
          <h2>Chat with AI</h2>
          <div className="chat-container" ref={chatContainerRef}>
            {chatHistory.map((chat, index) => (
              <div key={index} className={`chat-message ${chat.type}`}>
                <span className="chat-icon">
                  {chat.type === 'user' ? '👤' : chat.type === 'bot' ? '🤖' : '⚠️'}
                </span>
                <div className="chat-text">
                  <ReactMarkdown>{chat.message}</ReactMarkdown>
                </div>
              </div>
            ))}
            {chatLoading && (
              <div className="chat-message bot">
                <span className="chat-icon">🤖</span>
                <div className="chat-text">
                  <span className="loading-dots">Thinking</span>
                </div>
              </div>
            )}
          </div>
          <form onSubmit={handleChatSubmit} className="chat-input-container">
            <input
              type="text"
              value={chatMessage}
              onChange={(e) => setChatMessage(e.target.value)}
              placeholder="Ask about any species..."
              disabled={chatLoading}
              className="chat-input"
            />
            <button 
              type="submit" 
              disabled={chatLoading || !chatMessage.trim()}
              className="chat-submit"
            >
              Send
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

export default App;
