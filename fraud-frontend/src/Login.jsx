import { useState } from 'react'
import { login, register } from './api'

export default function Login({ onLoggedIn }) {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [isRegisterMode, setIsRegisterMode] = useState(false)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  async function handleSubmit(e) {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      if (isRegisterMode) {
        await register(email, password)
      }
      const result = await login(email, password)
      onLoggedIn(result.access_token)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <h1>Fraud Detection Platform</h1>
      <p className="subtitle">
        {isRegisterMode ? 'Create an account' : 'Sign in to score transactions'}
      </p>

      <div className="card">
        <form onSubmit={handleSubmit}>
          <label>Email</label>
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="you@example.com"
            required
          />

          <label>Password</label>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="********"
            required
          />

          {error && <div className="error">{error}</div>}

          <button type="submit" disabled={loading}>
            {loading ? 'Please wait...' : isRegisterMode ? 'Register & Sign In' : 'Sign In'}
          </button>
        </form>

        <p style={{ marginTop: 16, fontSize: 13, color: '#9aa0a8' }}>
          {isRegisterMode ? 'Already have an account? ' : "Don't have an account? "}
          <span
            className="logout-link"
            onClick={() => {
              setIsRegisterMode(!isRegisterMode)
              setError('')
            }}
          >
            {isRegisterMode ? 'Sign in' : 'Register'}
          </span>
        </p>
      </div>
    </div>
  )
}