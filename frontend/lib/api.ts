/**
 * MatGPT API client — talks to the local FastAPI server on port 8000.
 */

const BASE = 'http://localhost:8000'

export interface Model {
  id: string
}

export interface ConversationMeta {
  id: string
  name: string
  message_count: number
  updated_at: string
}

export interface Message {
  role: 'system' | 'user' | 'assistant'
  content: string
  token_count: number | null
}

export interface Conversation {
  id: string
  name: string
  messages: Message[]
}

export interface ChatResponse {
  response: string
  tokens: { system: number; messages: number; total: number }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}))
    throw new Error(detail?.detail ?? `HTTP ${res.status}`)
  }
  return res.json() as Promise<T>
}

export const api = {
  /** List models currently loaded in LM Studio */
  models: () =>
    request<{ models: string[] }>('/models'),

  /** Set the active model */
  selectModel: (model: string) =>
    request<{ selected: string }>('/models/select', {
      method: 'POST',
      body: JSON.stringify({ model }),
    }),

  /** List all saved conversations */
  conversations: () =>
    request<{ conversations: ConversationMeta[] }>('/conversations'),

  /** Create a new conversation */
  newConversation: (name: string, system_prompt = 'You are a helpful, concise assistant.') =>
    request<{ id: string; name: string }>('/conversations', {
      method: 'POST',
      body: JSON.stringify({ name, system_prompt }),
    }),

  /** Load a conversation with full message history */
  getConversation: (id: string) =>
    request<Conversation>(`/conversations/${id}`),

  /** Send a message and get a response */
  chat: (conv_id: string, message: string) =>
    request<ChatResponse>(`/conversations/${conv_id}/chat`, {
      method: 'POST',
      body: JSON.stringify({ message }),
    }),

  /** Delete a conversation */
  deleteConversation: (id: string) =>
    request<{ deleted: string }>(`/conversations/${id}`, { method: 'DELETE' }),

  /** Stream a chat response, calling onChunk for each text piece. */
  chatStream: async (
    conv_id: string,
    message: string,
    onChunk: (chunk: string) => void,
    thinking = false,
    signal?: AbortSignal,
  ): Promise<ChatResponse> => {
    const res = await fetch(`${BASE}/conversations/${conv_id}/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, thinking }),
      signal,
    })
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}))
      throw new Error(detail?.detail ?? `HTTP ${res.status}`)
    }

    const reader = res.body!.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    let finalTokens: ChatResponse['tokens'] = { system: 0, messages: 0, total: 0 }

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() ?? ''
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        const data = JSON.parse(line.slice(6))
        if (data.chunk) onChunk(data.chunk)
        if (data.done) finalTokens = data.tokens
        if (data.error) throw new Error(data.error)
      }
    }

    return { response: '', tokens: finalTokens }
  },
}
