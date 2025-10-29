import { useState } from 'react';
import { Send, Database, Search } from 'lucide-react';

export default function App() {
  const [question, setQuestion] = useState('');
  const [topK, setTopK] = useState(3);
  const [execute, setExecute] = useState(false);
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState(null);
  const [error, setError] = useState('');
  const [activeTab, setActiveTab] = useState('query');

  const API_URL = 'http://localhost:8080/api';

  const handleQuery = async () => {
    if (!question.trim()) return;
    
    setLoading(true);
    setError('');
    setResponse(null);

    try {
      const res = await fetch(`${API_URL}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, execute, topK })
      });
      
      const data = await res.json();
      setResponse(data);
    } catch (err) {
      setError('Failed to connect to backend: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleBuildSchema = async () => {
    setLoading(true);
    setError('');

    try {
      const res = await fetch(`${API_URL}/build-schema`, { method: 'POST' });
      const text = await res.text();
      alert(text);
    } catch (err) {
      setError('Failed to build schema: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleRetrieveSchema = async () => {
    if (!question.trim()) return;
    
    setLoading(true);
    setError('');
    setResponse(null);

    try {
      const res = await fetch(`${API_URL}/retrieve-schema`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, topK })
      });
      
      const text = await res.text();
      setResponse({ schema: text });
    } catch (err) {
      setError('Failed to retrieve schema: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-lg shadow-sm p-6 mb-6">
          <h1 className="text-2xl font-semibold text-gray-800 mb-2">SQL Query Interface</h1>
          <p className="text-gray-600 text-sm">Natural language to SQL conversion</p>
        </div>

        {/* Main Card */}
        <div className="bg-white rounded-lg shadow-sm p-6 mb-6">
          {/* Tabs */}
          <div className="flex gap-2 mb-6 border-b">
            <button
              onClick={() => setActiveTab('query')}
              className={`px-4 py-2 font-medium transition-colors ${
                activeTab === 'query' 
                  ? 'text-blue-600 border-b-2 border-blue-600' 
                  : 'text-gray-500 hover:text-gray-700'
              }`}
            >
              Query
            </button>
            <button
              onClick={() => setActiveTab('schema')}
              className={`px-4 py-2 font-medium transition-colors ${
                activeTab === 'schema' 
                  ? 'text-blue-600 border-b-2 border-blue-600' 
                  : 'text-gray-500 hover:text-gray-700'
              }`}
            >
              Schema
            </button>
          </div>

          {/* Query Tab */}
          {activeTab === 'query' && (
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Your Question
                </label>
                <textarea
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="e.g., Show me all customers from New York"
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                  rows="3"
                />
              </div>

              <div className="flex gap-4 items-center">
                <div className="flex-1">
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Top K Results
                  </label>
                  <input
                    type="number"
                    value={topK}
                    onChange={(e) => setTopK(parseInt(e.target.value) || 3)}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    min="1"
                  />
                </div>

                <div className="flex items-center pt-7">
                  <label className="flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={execute}
                      onChange={(e) => setExecute(e.target.checked)}
                      className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                    />
                    <span className="ml-2 text-sm font-medium text-gray-700">Execute Query</span>
                  </label>
                </div>
              </div>

              <button
                onClick={handleQuery}
                disabled={loading || !question.trim()}
                className="w-full bg-blue-600 text-white py-3 px-6 rounded-lg hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed flex items-center justify-center gap-2 font-medium transition-colors"
              >
                {loading ? (
                  <>Processing...</>
                ) : (
                  <>
                    <Send size={18} />
                    Generate SQL
                  </>
                )}
              </button>
            </div>
          )}

          {/* Schema Tab */}
          {activeTab === 'schema' && (
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Query for Schema
                </label>
                <textarea
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="Describe what you're looking for..."
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                  rows="3"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Top K Results
                </label>
                <input
                  type="number"
                  value={topK}
                  onChange={(e) => setTopK(parseInt(e.target.value) || 3)}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  min="1"
                />
              </div>

              <div className="flex gap-3">
                <button
                  onClick={handleRetrieveSchema}
                  disabled={loading || !question.trim()}
                  className="flex-1 bg-green-600 text-white py-3 px-6 rounded-lg hover:bg-green-700 disabled:bg-gray-300 disabled:cursor-not-allowed flex items-center justify-center gap-2 font-medium transition-colors"
                >
                  {loading ? (
                    <>Processing...</>
                  ) : (
                    <>
                      <Search size={18} />
                      Retrieve Schema
                    </>
                  )}
                </button>

                <button
                  onClick={handleBuildSchema}
                  disabled={loading}
                  className="flex-1 bg-purple-600 text-white py-3 px-6 rounded-lg hover:bg-purple-700 disabled:bg-gray-300 disabled:cursor-not-allowed flex items-center justify-center gap-2 font-medium transition-colors"
                >
                  {loading ? (
                    <>Processing...</>
                  ) : (
                    <>
                      <Database size={18} />
                      Build Schema
                    </>
                  )}
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Error Display */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
            <p className="text-red-800 text-sm">{error}</p>
          </div>
        )}

        {/* Response Display */}
        {response && (
          <div className="bg-white rounded-lg shadow-sm p-6">
            <h2 className="text-lg font-semibold text-gray-800 mb-4">Response</h2>
            
            {response.sql && (
              <div className="mb-4">
                <h3 className="text-sm font-medium text-gray-700 mb-2">Generated SQL</h3>
                <pre className="bg-gray-50 p-4 rounded-lg overflow-x-auto text-sm border border-gray-200">
                  <code>{response.sql}</code>
                </pre>
              </div>
            )}

            {response.executed && response.resultsText && (
              <div className="mb-4">
                <h3 className="text-sm font-medium text-gray-700 mb-2">Query Results</h3>
                <pre className="bg-gray-50 p-4 rounded-lg overflow-x-auto text-sm border border-gray-200">
                  <code>{response.resultsText}</code>
                </pre>
              </div>
            )}

            {response.schema && (
              <div>
                <h3 className="text-sm font-medium text-gray-700 mb-2">Schema Information</h3>
                <pre className="bg-gray-50 p-4 rounded-lg overflow-x-auto text-sm border border-gray-200">
                  <code>{response.schema}</code>
                </pre>
              </div>
            )}

            {response.error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                <p className="text-red-800 text-sm">{response.error}</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}