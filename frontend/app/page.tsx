'use client'

import { FormEvent, useCallback, useEffect, useRef, useState } from 'react'
import { ArrowUp, ChevronDown, Moon, Paperclip, Plus, Search, Sparkles, Sun, Trash2, Volume2 } from 'lucide-react'
import { api, ConversationMeta, Message } from '../lib/api'
import { fetchWeather, Weather } from '../lib/weather'

// ── Types ────────────────────────────────────────────────────────────────────

interface UiMessage {
  role: 'user' | 'assistant'
  text: string
  time: string
}

// ── Sub-components ───────────────────────────────────────────────────────────

function PixelLogo() {
  return (
    <div className="pixel-logo" aria-label="MatGPT logo" role="img">
      <div className="pixel-hair" />
      <div className="pixel-face">
        <span className="pixel-ear" />
        <span className="pixel-glasses left" />
        <span className="pixel-glasses right" />
        <span className="pixel-nose" />
        <span className="pixel-mouth" />
      </div>
    </div>
  )
}

function PixelScene({ night, weather }: { night: boolean; weather: Weather }) {
  return (
    <div className={`pixel-scene ${night ? 'night' : 'day'} ${weather}`} aria-hidden="true">
      <div className="pixel-sun" />
      <div className="pixel-moon" />
      <div className="pixel-cloud cloud-one" />
      <div className="pixel-cloud cloud-two" />
      <div className="pixel-wind wind-one" />
      <div className="pixel-wind wind-two" />
      <div className="pixel-bird bird-one">⌁</div>
      <div className="pixel-bird bird-two">⌁</div>
      <div className="pixel-bat bat-one">⌁</div>
      <div className="pixel-bat bat-two">⌁</div>
      <div className="pixel-rain rain-one" /><div className="pixel-rain rain-two" /><div className="pixel-rain rain-three" />
      <div className="pixel-star star-one">✦</div>
      <div className="pixel-star star-two">✦</div>
      <div className="pixel-house house-one"><i /><b /><em /></div>
      <div className="pixel-house house-two"><i /><b /><em /></div>
      <div className="pixel-hill back" />
      <div className="pixel-hill front" />
      <div className="pixel-tree tree-one"><i /><b /></div>
      <div className="pixel-tree tree-two"><i /><b /></div>
      <div className="pixel-ground" />
    </div>
  )
}

function TypingIndicator() {
  return (
    <div className="message-row assistant">
      <div className="message-avatar">m</div>
      <div className="message-copy">
        <div className="message-meta"><strong>matgpt</strong></div>
        <p className="typing-dots"><span /><span /><span /></p>
      </div>
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function Page() {
  // Time / scene
  const [night, setNight] = useState(() => {
    const h = new Date().getHours()
    return h < 7 || h >= 19
  })
  const [weather, setWeather] = useState<Weather>('clear')

  // Backend state
  const [models, setModels]           = useState<string[]>([])
  const [activeModel, setActiveModel] = useState<string>('')
  const [history, setHistory]         = useState<ConversationMeta[]>([])
  const [convId, setConvId]           = useState<string | null>(null)
  const [convName, setConvName]       = useState('New conversation')

  // Chat state
  const [messages, setMessages] = useState<UiMessage[]>([])
  const [draft, setDraft]       = useState('')
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState<string | null>(null)
  const [tokens, setTokens]     = useState(0)

  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef       = useRef<HTMLInputElement>(null)

  // ── Boot ─────────────────────────────────────────────────────────────────

  useEffect(() => {
    // Weather (non-blocking)
    fetchWeather().then(setWeather)

    // Backend
    async function boot() {
      try {
        const [{ models: ms }, { conversations: convs }] = await Promise.all([
          api.models(),
          api.conversations(),
        ])
        setModels(ms)
        if (ms.length) {
          setActiveModel(ms[0])
          await api.selectModel(ms[0])
        }
        setHistory(convs)
      } catch {
        setError('Cannot reach backend. Run: python -m matgpt.server')
      }
    }
    boot()
  }, [])

  // Scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  // ── Handlers ─────────────────────────────────────────────────────────────

  const now = () =>
    new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })

  const refreshHistory = useCallback(async () => {
    try {
      const { conversations } = await api.conversations()
      setHistory(conversations)
    } catch {}
  }, [])

  async function submitMessage(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const text = draft.trim()
    if (!text || loading) return

    setDraft('')
    setError(null)
    setLoading(true)
    setMessages((prev) => [...prev, { role: 'user', text, time: now() }])

    try {
      // Create conversation on first message
      let id = convId
      if (!id) {
        const conv = await api.newConversation(text.slice(0, 50))
        id = conv.id
        setConvId(id)
        setConvName(conv.name)
      }

      const data = await api.chat(id, text)

      setMessages((prev) => [
        ...prev,
        { role: 'assistant', text: data.response, time: now() },
      ])
      setTokens(data.tokens.total)
      refreshHistory()
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Something went wrong.'
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', text: `Error: ${msg}`, time: now() },
      ])
    } finally {
      setLoading(false)
      inputRef.current?.focus()
    }
  }

  async function handleSelectModel(model: string) {
    setActiveModel(model)
    try { await api.selectModel(model) } catch {}
  }

  async function openConversation(id: string, name: string) {
    setConvId(id)
    setConvName(name)
    setError(null)
    try {
      const conv = await api.getConversation(id)
      const uiMsgs: UiMessage[] = conv.messages
        .filter((m) => m.role !== 'system')
        .map((m) => ({
          role: m.role as 'user' | 'assistant',
          text: m.content,
          time: '',
        }))
      setMessages(uiMsgs)
      const total = conv.messages.reduce((s, m) => s + (m.token_count ?? 0), 0)
      setTokens(total)
    } catch {}
    refreshHistory()
  }

  async function deleteConversation(e: React.MouseEvent, id: string) {
    e.stopPropagation()
    await api.deleteConversation(id)
    if (convId === id) startNewChat()
    refreshHistory()
  }

  function startNewChat() {
    setConvId(null)
    setConvName('New conversation')
    setMessages([])
    setTokens(0)
    setError(null)
    inputRef.current?.focus()
  }

  // ── Render ────────────────────────────────────────────────────────────────

  const greeting = night ? 'Good evening, Mat.' : 'Good morning, Mat.'
  const isEmpty  = messages.length === 0

  return (
    <main className={`app-shell ${night ? 'is-night' : 'is-day'}`}>
      <PixelScene night={night} weather={weather} />
      <div className="grain" />

      <section className="app-window" aria-label="MatGPT chatbot">

        {/* ── Sidebar ── */}
        <aside className="sidebar">
          <div className="brand-row">
            <PixelLogo />
            <div>
              <div className="brand-name">mat<span>gpt</span></div>
              <div className="brand-status">
                <span />
                {activeModel
                  ? activeModel.split('/').pop()?.split('-').slice(0, 2).join('-')
                  : 'connecting…'}
              </div>
            </div>
          </div>

          <button className="new-chat" onClick={startNewChat}>
            <Plus size={15} /> New chat <span>⌘ K</span>
          </button>

          <div className="sidebar-label">Recent</div>

          {history.length === 0 && (
            <p style={{ fontSize: 10, color: '#879487', padding: '6px 10px' }}>
              No conversations yet.
            </p>
          )}

          {history.map((conv) => (
            <button
              key={conv.id}
              className={`history-item ${conv.id === convId ? 'active' : ''}`}
              onClick={() => openConversation(conv.id, conv.name)}
              title={conv.name}
            >
              <span className="history-dot" />
              <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {conv.name}
              </span>
              <span
                className="history-delete"
                onClick={(e) => deleteConversation(e, conv.id)}
                title="Delete"
                role="button"
              >
                <Trash2 size={11} />
              </span>
            </button>
          ))}

          <div className="sidebar-bottom">
            {/* Model selector */}
            {models.length > 0 && (
              <select
                className="model-select"
                value={activeModel}
                onChange={(e) => handleSelectModel(e.target.value)}
                aria-label="Select model"
              >
                {models.map((m) => (
                  <option key={m} value={m}>{m.split('/').pop()}</option>
                ))}
              </select>
            )}

            <button className="side-action"><Search size={15} /> Search chats <kbd>⌘ F</kbd></button>
            <button className="side-action"><Volume2 size={15} /> Sound &amp; voice</button>
            <div className="profile">
              <div className="profile-avatar">M</div>
              <div><strong>Mat</strong><small>Personal workspace</small></div>
              <ChevronDown size={14} />
            </div>
          </div>
        </aside>

        {/* ── Chat area ── */}
        <div className="chat-area">
          <header className="chat-header">
            <div>
              <span className="eyebrow">A little space to think</span>
              <h1>{convName}</h1>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              {tokens > 0 && (
                <span className="token-badge">{tokens.toLocaleString()} tokens</span>
              )}
              <button
                className="mode-toggle"
                onClick={() => setNight(!night)}
                aria-label={`Switch to ${night ? 'day' : 'night'} mode`}
              >
                {night ? <Moon size={15} /> : <Sun size={15} />}
                <span>{night ? 'Night' : 'Day'}</span>
              </button>
            </div>
          </header>

          <div className="conversation">
            <div className="date-line"><span>Today</span></div>

            {isEmpty && (
              <div className="welcome-block">
                <div className="mini-mark"><Sparkles size={16} /></div>
                <p>{greeting}</p>
                <span>Take your time. I&apos;m here when you are.</span>
              </div>
            )}

            <div className="message-list">
              {messages.map((message, index) => (
                <div className={`message-row ${message.role}`} key={index}>
                  <div className="message-avatar">
                    {message.role === 'assistant' ? 'm' : 'M'}
                  </div>
                  <div className="message-copy">
                    <div className="message-meta">
                      <strong>{message.role === 'assistant' ? 'matgpt' : 'you'}</strong>
                      {message.time && <time>{message.time}</time>}
                    </div>
                    <p style={{ whiteSpace: 'pre-wrap' }}>{message.text}</p>
                    {message.role === 'assistant' && (
                      <button className="listen"><Volume2 size={13} /> Listen</button>
                    )}
                  </div>
                </div>
              ))}

              {loading && <TypingIndicator />}

              {error && (
                <div className="message-row assistant">
                  <div className="message-avatar">!</div>
                  <div className="message-copy">
                    <p style={{ color: '#c0392b', fontSize: 12 }}>{error}</p>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          </div>

          <form className="composer" onSubmit={submitMessage}>
            <button type="button" className="icon-button" aria-label="Attach file">
              <Paperclip size={18} />
            </button>
            <input
              ref={inputRef}
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              placeholder="Ask matgpt anything…"
              aria-label="Message matgpt"
              disabled={loading}
              autoFocus
            />
            <span className="enter-hint">↵ send</span>
            <button type="submit" className="send-button" aria-label="Send" disabled={loading || !draft.trim()}>
              <ArrowUp size={17} />
            </button>
          </form>

          <footer className="chat-footer">
            <span>matgpt can make mistakes. Check important info.</span>
            <span className="keyboard-hint">⌘ + K <b>new chat</b></span>
          </footer>
        </div>
      </section>
    </main>
  )
}
