# Project 1 — Local AI Assistant
### Ship Date: End of Month 1 | Stack: Next.js + Hono + LangChain.js + Ollama + ChromaDB

---

## What You're Building

A personal AI assistant that runs 100% offline on your Mac.

Features:
- Clean chat UI built in Next.js
- Powered by Mistral 7B via Ollama
- Three modes: Chat, Agent (with tools), Document Q&A (RAG)
- Streams responses in real-time
- Remembers conversation history

This is your first portfolio project. It proves you can build a full-stack AI app
from scratch with zero cloud dependency.

---

## Project Structure

```
Project1_LocalAssistant/
├── backend/
│   ├── server.ts          ← Hono server (all routes)
│   ├── agent.ts           ← Agent with tools
│   ├── rag.ts             ← RAG chain + retriever
│   └── package.json
└── frontend/
    ├── app/
    │   ├── page.tsx        ← Main chat UI
    │   └── layout.tsx
    ├── components/
    │   ├── ChatMessage.tsx  ← Single message bubble
    │   ├── AgentSteps.tsx   ← Tool call visualizer
    │   └── ModeSelector.tsx ← Switch between Chat/Agent/RAG
    └── package.json
```

---

## Setup

```bash
cd "/Users/b0268804/Documents/untitled folder/Claude/AgenticAI/Month1/Project1_LocalAssistant"
mkdir backend frontend
```

### Backend setup:
```bash
cd backend
pnpm init -y
pnpm add hono @hono/node-server @langchain/ollama @langchain/core @langchain/community langchain chromadb zod
pnpm add -D typescript tsx @types/node
```

### Frontend setup:
```bash
cd ../frontend
npx create-next-app@latest . --typescript --tailwind --app --no-src-dir
```

---

## Backend — server.ts

```typescript
import { Hono } from "hono";
import { serve } from "@hono/node-server";
import { cors } from "hono/cors";
import { streamText } from "hono/streaming";
import { ChatOllama } from "@langchain/ollama";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { agentExecutor } from "./agent";
import { ragChain, retriever, indexDocument } from "./rag";

const app = new Hono();
app.use("*", cors({ origin: "http://localhost:3000" }));

const llm = new ChatOllama({ model: "mistral:7b" });

interface Message { role: "user" | "assistant"; content: string; }

// ── Route 1: Streaming Chat ────────────────────────────────────────────────
app.post("/chat", async (c) => {
  const { messages, systemPrompt = "You are a helpful AI assistant." } =
    await c.req.json<{ messages: Message[]; systemPrompt?: string }>();

  const langchainMsgs = [
    new SystemMessage(systemPrompt),
    ...messages.filter((m) => m.role === "user").map((m) => new HumanMessage(m.content)),
  ];

  return streamText(c, async (stream) => {
    for await (const chunk of await llm.stream(langchainMsgs)) {
      if (chunk.content) {
        await stream.write(`data: ${JSON.stringify({ token: chunk.content })}\n\n`);
      }
    }
    await stream.write("data: [DONE]\n\n");
  });
});

// ── Route 2: Agent ─────────────────────────────────────────────────────────
app.post("/agent", async (c) => {
  const { input } = await c.req.json<{ input: string }>();
  const result = await agentExecutor.invoke({ input });

  return c.json({
    output: result.output,
    steps: result.intermediateSteps?.map((step: any) => ({
      tool: step.action.tool,
      input: step.action.toolInput,
      output: step.observation,
    })) ?? [],
  });
});

// ── Route 3: RAG ───────────────────────────────────────────────────────────
app.post("/rag", async (c) => {
  const { question } = await c.req.json<{ question: string }>();
  const docs = await retriever.invoke(question);
  const answer = await ragChain.invoke(question);

  return c.json({
    answer,
    sources: docs.map((d) => ({
      content: d.pageContent,
      source: d.metadata.source ?? "unknown",
    })),
  });
});

// ── Route 4: Index a document ──────────────────────────────────────────────
app.post("/index", async (c) => {
  const { filePath } = await c.req.json<{ filePath: string }>();
  await indexDocument(filePath);
  return c.json({ status: "indexed", filePath });
});

app.get("/health", (c) => c.json({ status: "ok" }));

serve({ fetch: app.fetch, port: 8000 }, () =>
  console.log("Project 1 backend running at http://localhost:8000")
);
```

---

## Backend — agent.ts

```typescript
import { ChatOllama } from "@langchain/ollama";
import { tool } from "@langchain/core/tools";
import { createReactAgent, AgentExecutor } from "langchain/agents";
import { z } from "zod";

const calculatorTool = tool(
  async ({ expression }) => {
    try { return String(new Function(`return ${expression}`)()) }
    catch (e) { return `Error: ${e}` }
  },
  {
    name: "calculator",
    description: "Evaluates a JS math expression. Use for any calculations.",
    schema: z.object({ expression: z.string() }),
  }
);

const weatherTool = tool(
  async ({ city }) => {
    const res = await fetch(`https://wttr.in/${city}?format=3`);
    return res.text();
  },
  {
    name: "get_weather",
    description: "Gets current weather for a city. Input: city name only.",
    schema: z.object({ city: z.string() }),
  }
);

const tools = [calculatorTool, weatherTool];
const llm = new ChatOllama({ model: "mistral:7b", temperature: 0 });

export const agentExecutor = new AgentExecutor({
  agent: await createReactAgent({ llm: llm.bindTools(tools), tools }),
  tools,
  verbose: false,
  maxIterations: 5,
  handleParsingErrors: true,
});
```

---

## Backend — rag.ts

```typescript
import { OllamaEmbeddings, ChatOllama } from "@langchain/ollama";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnablePassthrough, RunnableSequence } from "@langchain/core/runnables";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import type { Document } from "@langchain/core/documents";
import { readFileSync } from "fs";

const embeddings = new OllamaEmbeddings({ model: "nomic-embed-text" });
const llm = new ChatOllama({ model: "mistral:7b" });
const CHROMA_URL = "http://localhost:8001";
const COLLECTION = "assistant_docs";

const vectorstore = await Chroma.fromExistingCollection(embeddings, {
  collectionName: COLLECTION,
  url: CHROMA_URL,
}).catch(() => null);

export const retriever = vectorstore?.asRetriever({ k: 3 });

const prompt = ChatPromptTemplate.fromTemplate(`
Answer based ONLY on the context below. If not found, say so.

Context: {context}
Question: {question}
Answer:`);

const formatDocs = (docs: Document[]) =>
  docs.map((d) => d.pageContent).join("\n\n---\n\n");

export const ragChain = RunnableSequence.from([
  { context: retriever!.pipe(formatDocs), question: new RunnablePassthrough() },
  prompt,
  llm,
  new StringOutputParser(),
]);

export async function indexDocument(filePath: string) {
  const text = readFileSync(filePath, "utf-8");
  const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 500, chunkOverlap: 50 });
  const docs = await splitter.createDocuments([text], [{ source: filePath }]);

  await Chroma.fromDocuments(docs, embeddings, {
    collectionName: COLLECTION,
    url: CHROMA_URL,
  });
}
```

---

## Frontend — app/page.tsx

```tsx
"use client";

import { useState } from "react";
import ChatMessage from "@/components/ChatMessage";
import AgentSteps from "@/components/AgentSteps";
import ModeSelector from "@/components/ModeSelector";

type Mode = "chat" | "agent" | "rag";

interface Message {
  role: "user" | "assistant";
  content: string;
  steps?: { tool: string; input: any; output: string }[];
  sources?: { content: string; source: string }[];
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [mode, setMode] = useState<Mode>("chat");
  const [isLoading, setIsLoading] = useState(false);

  const send = async () => {
    if (!input.trim() || isLoading) return;
    const userMsg: Message = { role: "user", content: input };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setIsLoading(true);

    if (mode === "chat") {
      setMessages((prev) => [...prev, { role: "assistant", content: "" }]);
      const res = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: [...messages, userMsg] }),
      });
      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        for (const line of decoder.decode(value).split("\n")) {
          if (!line.startsWith("data: ")) continue;
          const data = line.slice(6);
          if (data === "[DONE]") break;
          try {
            const { token } = JSON.parse(data);
            setMessages((prev) => {
              const updated = [...prev];
              updated[updated.length - 1].content += token;
              return updated;
            });
          } catch {}
        }
      }
    } else if (mode === "agent") {
      const res = await fetch("http://localhost:8000/agent", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ input: userMsg.content }),
      });
      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.output, steps: data.steps },
      ]);
    } else {
      const res = await fetch("http://localhost:8000/rag", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: userMsg.content }),
      });
      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.answer, sources: data.sources },
      ]);
    }

    setIsLoading(false);
  };

  return (
    <div className="flex flex-col h-screen bg-gray-950 text-white">
      <div className="border-b border-gray-800 p-4 flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold">Local AI Assistant</h1>
          <p className="text-xs text-gray-500">100% local • Mistral 7B • No API cost</p>
        </div>
        <ModeSelector mode={mode} onChange={setMode} />
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg, i) => (
          <div key={i}>
            <ChatMessage message={msg} />
            {msg.steps && <AgentSteps steps={msg.steps} />}
            {msg.sources && (
              <div className="text-xs text-gray-500 mt-1 pl-2">
                Sources: {msg.sources.map((s) => s.source).join(", ")}
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="border-t border-gray-800 p-4">
        <div className="flex gap-3 max-w-3xl mx-auto">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
            placeholder={mode === "rag" ? "Ask about your documents..." : "Type a message..."}
            rows={1}
            className="flex-1 bg-gray-800 rounded-xl px-4 py-3 text-sm resize-none outline-none focus:ring-1 focus:ring-blue-500"
          />
          <button
            onClick={send}
            disabled={isLoading || !input.trim()}
            className="bg-blue-600 hover:bg-blue-700 disabled:opacity-40 px-5 py-3 rounded-xl text-sm font-medium"
          >
            {isLoading ? "..." : "Send"}
          </button>
        </div>
      </div>
    </div>
  );
}
```

---

## Frontend Components

### components/ChatMessage.tsx
```tsx
interface Message {
  role: "user" | "assistant";
  content: string;
}

export default function ChatMessage({ message }: { message: Message }) {
  return (
    <div className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[75%] rounded-2xl px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap ${
          message.role === "user" ? "bg-blue-600" : "bg-gray-800 text-gray-100"
        }`}
      >
        {message.content}
      </div>
    </div>
  );
}
```

### components/AgentSteps.tsx
```tsx
interface Step { tool: string; input: any; output: string; }

export default function AgentSteps({ steps }: { steps: Step[] }) {
  if (!steps.length) return null;
  return (
    <div className="ml-4 mt-1 bg-gray-900 rounded-xl p-3 text-xs space-y-2">
      <p className="text-gray-400 font-medium">Agent used tools:</p>
      {steps.map((step, i) => (
        <div key={i} className="border-l-2 border-blue-500 pl-3 space-y-1">
          <p className="text-blue-400">🔧 {step.tool}</p>
          <p className="text-gray-400">Input: {JSON.stringify(step.input)}</p>
          <p className="text-green-400">Result: {step.output}</p>
        </div>
      ))}
    </div>
  );
}
```

### components/ModeSelector.tsx
```tsx
type Mode = "chat" | "agent" | "rag";

const modes: { value: Mode; label: string; emoji: string }[] = [
  { value: "chat", label: "Chat", emoji: "💬" },
  { value: "agent", label: "Agent", emoji: "🤖" },
  { value: "rag", label: "Docs", emoji: "📄" },
];

export default function ModeSelector({
  mode,
  onChange,
}: {
  mode: Mode;
  onChange: (m: Mode) => void;
}) {
  return (
    <div className="flex gap-1 bg-gray-800 rounded-xl p-1">
      {modes.map((m) => (
        <button
          key={m.value}
          onClick={() => onChange(m.value)}
          className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
            mode === m.value ? "bg-blue-600 text-white" : "text-gray-400 hover:text-white"
          }`}
        >
          {m.emoji} {m.label}
        </button>
      ))}
    </div>
  );
}
```

---

## Running Project 1

You need 3 terminal tabs:

Tab 1 — Ollama:
```bash
ollama serve
```

Tab 2 — ChromaDB (only needed for RAG mode):
```bash
chroma run --port 8001
```

Tab 3 — Backend:
```bash
cd Project1_LocalAssistant/backend
tsx server.ts
```

Tab 4 — Frontend:
```bash
cd Project1_LocalAssistant/frontend
npm run dev
```

Open http://localhost:3000

---

## What to Demo

1. Chat mode — have a normal conversation
2. Agent mode — ask "What is 15% of 2500?" and show the tool call visualization
3. Agent mode — ask "What's the weather in Tokyo?"
4. RAG mode — index a document, then ask questions about it

---

## Next → [Month 2: Week 5 — Multi-Agent Systems](../../Month2/Week5_MultiAgent/notes.md)
