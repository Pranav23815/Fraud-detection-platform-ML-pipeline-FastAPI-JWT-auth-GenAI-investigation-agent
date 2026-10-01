import { useState } from 'react'
import { predict } from './api'

function randomAccountId(prefix) {
  const digits = Math.floor(1000000000 + Math.random() * 9000000000)
  return `${prefix}${digits}`
}

function generateFraudExample() {
  // Fraud pattern: drains almost the entire account balance in one TRANSFER
  const balance = Math.floor(100 + Math.random() * 9900)
  const amount = Math.round(balance * (0.9 + Math.random() * 0.1) * 100) / 100
  return {
    step: Math.floor(1 + Math.random() * 48),
    type: 'TRANSFER',
    amount,
    name_orig: randomAccountId('C'),
    oldbalance_org: balance,
    name_dest: randomAccountId('C'),
    oldbalance_dest: 0,
  }
}

function generateLegitExample() {
  // Legit pattern: small payment relative to a healthy account balance, to a merchant
  const balance = Math.floor(1000 + Math.random() * 9000)
  const amount = Math.round((10 + Math.random() * 200) * 100) / 100
  return {
    step: Math.floor(1 + Math.random() * 48),
    type: 'PAYMENT',
    amount,
    name_orig: randomAccountId('C'),
    oldbalance_org: balance,
    name_dest: randomAccountId('M'),
    oldbalance_dest: 0,
  }
}

export default function Dashboard({ token, onLogout }) {
  const [form, setForm] = useState(generateFraudExample)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  function updateField(field, value) {
    setForm({ ...form, [field]: value })
  }

  async function handleSubmit(e) {
    e.preventDefault()
    setError('')
    setLoading(true)
    setResult(null)
    try {
      const res = await predict(token, {
        ...form,
        step: Number(form.step),
        amount: Number(form.amount),
        oldbalance_org: Number(form.oldbalance_org),
        oldbalance_dest: Number(form.oldbalance_dest),
      })
      setResult(res)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const maxAbsShap = result
    ? Math.max(...result.top_contributing_features.map((f) => Math.abs(f.shap_contribution)))
    : 1

  return (
    <div className="app">
      <div className="top-bar">
        <div>
          <h1>Fraud Detection Platform</h1>
          <p className="subtitle" style={{ marginBottom: 0 }}>
            Real-time transaction scoring
          </p>
        </div>
        <span className="logout-link" onClick={onLogout}>
          Log out
        </span>
      </div>

      <div className="card">
        <form onSubmit={handleSubmit}>
          <label>Transaction Type</label>
          <select value={form.type} onChange={(e) => updateField('type', e.target.value)}>
            <option value="TRANSFER">TRANSFER</option>
            <option value="CASH_OUT">CASH_OUT</option>
            <option value="PAYMENT">PAYMENT</option>
            <option value="CASH_IN">CASH_IN</option>
            <option value="DEBIT">DEBIT</option>
          </select>

          <label>Amount</label>
          <input
            type="number"
            step="any"
            value={form.amount}
            onChange={(e) => updateField('amount', e.target.value)}
            required
          />

          <label>Origin Account Balance (before)</label>
          <input
            type="number"
            step="any"
            value={form.oldbalance_org}
            onChange={(e) => updateField('oldbalance_org', e.target.value)}
            required
          />

          <label>Destination Account Balance (before)</label>
          <input
            type="number"
            step="any"
            value={form.oldbalance_dest}
            onChange={(e) => updateField('oldbalance_dest', e.target.value)}
            required
          />

          <label>Origin Account ID</label>
          <input
            type="text"
            value={form.name_orig}
            onChange={(e) => updateField('name_orig', e.target.value)}
            required
          />

          <label>Destination Account ID</label>
          <input
            type="text"
            value={form.name_dest}
            onChange={(e) => updateField('name_dest', e.target.value)}
            required
          />

          {error && <div className="error">{error}</div>}

          <button type="submit" disabled={loading}>
            {loading ? 'Scoring transaction...' : 'Score Transaction'}
          </button>
        </form>

        <div style={{ marginTop: 16, display: 'flex', gap: 10 }}>
          <span className="logout-link" onClick={() => setForm(generateFraudExample())}>
            Load fraud-like example
          </span>
          <span className="logout-link" onClick={() => setForm(generateLegitExample())}>
            Load legit-like example
          </span>
        </div>
      </div>

      {result && (
        <div className="card">
          <div className="result-header">
            <div>
              <div style={{ fontSize: 12, color: '#9aa0a8', marginBottom: 4 }}>
                FRAUD PROBABILITY
              </div>
              <div className="probability">
                {(result.fraud_probability * 100).toFixed(2)}%
              </div>
            </div>
            <span className={`badge ${result.is_flagged ? 'flagged' : 'clear'}`}>
              {result.is_flagged ? 'FLAGGED' : 'CLEAR'}
            </span>
          </div>

          <div style={{ fontSize: 12, color: '#9aa0a8', marginBottom: 12, marginTop: 20 }}>
            TOP CONTRIBUTING FEATURES (SHAP)
          </div>
          {result.top_contributing_features.map((f) => (
            <div className="feature-row" key={f.feature}>
              <div className="feature-name">{f.feature}</div>
              <div className="feature-bar-track">
                <div
                  className={`feature-bar ${f.shap_contribution >= 0 ? 'positive' : 'negative'}`}
                  style={{
                    width: `${(Math.abs(f.shap_contribution) / maxAbsShap) * 100}%`,
                    marginLeft: f.shap_contribution < 0 ? 'auto' : 0,
                  }}
                />
              </div>
              <div style={{ width: 50, textAlign: 'right', color: '#9aa0a8' }}>
                {f.shap_contribution.toFixed(2)}
              </div>
            </div>
          ))}

          {result.investigation_report && (
            <>
              <div style={{ fontSize: 12, color: '#9aa0a8', marginBottom: 12, marginTop: 24 }}>
                GENAI INVESTIGATION REPORT
              </div>
              <div className="report-box">
                <span className={`risk-level ${result.investigation_report.risk_level}`}>
                  {result.investigation_report.risk_level}
                </span>
                <p style={{ fontSize: 14, lineHeight: 1.6, marginBottom: 10 }}>
                  {result.investigation_report.summary}
                </p>
                <p style={{ fontSize: 13, color: '#9aa0a8' }}>
                  <strong>Recommended action:</strong>{' '}
                  {result.investigation_report.recommended_action}
                </p>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  )
}