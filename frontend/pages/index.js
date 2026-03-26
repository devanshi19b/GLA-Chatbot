import Head from "next/head";
import axios from "axios";
import { useEffect, useRef, useState } from "react";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

const starterMessage = {
  role: "assistant",
  content:
    "Ask me anything from the GLA University brochures. If the brochures do not mention it, I will say so.",
  sources: [],
};

const exampleQuestions = [
  "What courses are mentioned in the brochure?",
  "What does the brochure say about placements?",
  "What facilities are available on campus?",
];

export default function Home() {
  const [messages, setMessages] = useState([starterMessage]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const bottomRef = useRef(null);
  const conversationCount = Math.max(messages.length - 1, 0);
  const trimmedInput = input.trim();

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const sendMessage = async () => {
    const trimmedInput = input.trim();
    if (!trimmedInput || loading) {
      return;
    }

    setError("");
    setLoading(true);
    setInput("");

    const userMessage = { role: "user", content: trimmedInput };
    setMessages((currentMessages) => [...currentMessages, userMessage]);

    try {
      const response = await axios.post(`${API_BASE_URL}/chat`, {
        message: trimmedInput,
      });

      setMessages((currentMessages) => [
        ...currentMessages,
        {
          role: "assistant",
          content: response.data.answer,
          sources: response.data.sources || [],
        },
      ]);
    } catch (requestError) {
      const detail =
        requestError?.response?.data?.detail ||
        "The chatbot could not reach the backend. Please check that the FastAPI server is running.";

      setError(detail);
      setMessages((currentMessages) => [
        ...currentMessages,
        {
          role: "assistant",
          content:
            "I couldn't fetch a brochure-grounded answer because the backend request failed. Please verify the API server is running and try again.",
          sources: [],
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = async (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      await sendMessage();
    }
  };

  const handleExampleClick = (question) => {
    setInput(question);
  };

  const clearChat = () => {
    if (loading) {
      return;
    }

    setMessages([starterMessage]);
    setInput("");
    setError("");
  };

  return (
    <>
      <Head>
        <title>GLA University Brochure Chatbot</title>
        <meta
          name="description"
          content="A RAG chatbot that answers only from the uploaded GLA University brochures."
        />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      <main className="min-h-screen px-4 py-8 sm:px-6 lg:px-8">
        <div className="mx-auto flex min-h-[calc(100vh-4rem)] max-w-6xl flex-col gap-6 lg:flex-row">
          <section className="relative overflow-hidden rounded-[32px] border border-white/60 bg-white/75 p-8 shadow-panel backdrop-blur xl:w-[38%]">
            <div className="absolute inset-x-0 top-0 h-1 bg-gradient-to-r from-brand.gold via-amber-500 to-sky-500" />
            <span className="inline-flex rounded-full bg-brand.blush px-3 py-1 text-xs font-semibold uppercase tracking-[0.24em] text-brand.ink/80">
              RAG Powered
            </span>
            <h1 className="mt-5 font-['Playfair_Display'] text-4xl font-semibold leading-tight text-brand.ink sm:text-5xl">
              GLA University brochure assistant
            </h1>
            <p className="mt-5 max-w-xl text-sm leading-7 text-brand.ink/75 sm:text-base">
              This chatbot only answers from the uploaded brochure PDFs. It does not use outside
              knowledge, and if the brochures do not cover something, it says that directly.
            </p>

            <div className="mt-8 grid gap-4 sm:grid-cols-2">
              <div className="rounded-3xl border border-brand.ink/10 bg-brand.mist p-4">
                <p className="text-xs font-semibold uppercase tracking-[0.2em] text-brand.ink/55">
                  Source control
                </p>
                <p className="mt-2 text-sm text-brand.ink/80">
                  Answers are grounded in brochure chunks retrieved from FAISS.
                </p>
              </div>
              <div className="rounded-3xl border border-brand.ink/10 bg-white p-4">
                <p className="text-xs font-semibold uppercase tracking-[0.2em] text-brand.ink/55">
                  Fallback behavior
                </p>
                <p className="mt-2 text-sm text-brand.ink/80">
                  If the answer is missing, the bot responds: Information not available in the
                  brochure.
                </p>
              </div>
            </div>

            <div className="mt-8">
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-brand.ink/55">
                Try these
              </p>
              <div className="mt-3 flex flex-wrap gap-3">
                {exampleQuestions.map((question) => (
                  <button
                    key={question}
                    type="button"
                    onClick={() => handleExampleClick(question)}
                    className="rounded-full border border-brand.ink/10 bg-white px-4 py-2 text-left text-sm text-brand.ink/80 transition hover:border-brand.gold/60 hover:text-brand.ink"
                  >
                    {question}
                  </button>
                ))}
              </div>
            </div>
          </section>

          <section className="flex min-h-[640px] flex-1 flex-col overflow-hidden rounded-[32px] border border-white/60 bg-slate-950/95 shadow-panel">
            <div className="border-b border-white/10 px-6 py-5">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <div className="flex flex-wrap items-center gap-3">
                    <p className="text-sm font-semibold uppercase tracking-[0.24em] text-amber-300/90">
                      Live chat
                    </p>
                    <span className="rounded-full border border-emerald-300/30 bg-emerald-400/10 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.18em] text-emerald-200">
                      {loading ? "Searching" : "Ready"}
                    </span>
                  </div>
                  <p className="mt-2 text-sm text-slate-300">
                    Ask about admissions, courses, facilities, placements, fees, or anything explicitly
                    present in the brochures.
                  </p>
                </div>

                <button
                  type="button"
                  onClick={clearChat}
                  disabled={loading}
                  className="rounded-full border border-white/15 px-4 py-2 text-xs font-semibold uppercase tracking-[0.18em] text-slate-200 transition hover:border-amber-300/60 hover:text-white disabled:cursor-not-allowed disabled:opacity-60"
                >
                  Clear chat
                </button>
              </div>
            </div>

            <div className="message-scrollbar flex-1 space-y-4 overflow-y-auto px-4 py-5 sm:px-6">
              <div className="flex items-center justify-between px-1 text-xs uppercase tracking-[0.18em] text-slate-400">
                <span>{conversationCount} messages</span>
                <span>Brochures only</span>
              </div>

              {messages.map((message, index) => (
                <div
                  key={`${message.role}-${index}`}
                  className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  <div
                    className={`max-w-2xl rounded-3xl px-4 py-3 text-sm leading-7 shadow-lg sm:text-[15px] ${
                      message.role === "user"
                        ? "rounded-br-md bg-gradient-to-r from-amber-400 to-orange-400 text-slate-950"
                        : "rounded-bl-md border border-white/10 bg-white/[0.08] text-slate-100"
                    }`}
                  >
                    <p className="whitespace-pre-wrap">{message.content}</p>
                    {message.role === "assistant" && message.sources?.length > 0 && (
                      <p className="mt-3 text-xs uppercase tracking-[0.18em] text-slate-400">
                        Source: {message.sources.join(", ")}
                      </p>
                    )}
                  </div>
                </div>
              ))}

              {loading && (
                <div className="flex justify-start">
                  <div className="max-w-sm rounded-3xl rounded-bl-md border border-white/10 bg-white/[0.08] px-4 py-3 text-sm text-slate-200">
                    Searching brochures and drafting a grounded answer...
                  </div>
                </div>
              )}

              <div ref={bottomRef} />
            </div>

            <div className="border-t border-white/10 bg-slate-950/95 px-4 py-4 sm:px-6">
              {error && (
                <p className="mb-3 rounded-2xl border border-red-400/30 bg-red-500/10 px-4 py-3 text-sm text-red-200">
                  {error}
                </p>
              )}

              <div className="flex flex-col gap-3 sm:flex-row">
                <div className="flex-1">
                  <textarea
                    value={input}
                    onChange={(event) => setInput(event.target.value)}
                    onKeyDown={handleKeyDown}
                    rows={3}
                    placeholder="Ask a question about the brochures..."
                    className="min-h-[72px] w-full resize-none rounded-3xl border border-white/10 bg-white/[0.08] px-4 py-3 text-sm text-white outline-none transition focus:border-amber-300/70 focus:ring-2 focus:ring-amber-300/20"
                  />
                  <div className="mt-2 flex items-center justify-between px-1 text-xs text-slate-400">
                    <span>Press Enter to send, Shift + Enter for a new line</span>
                    <span>{trimmedInput.length} chars</span>
                  </div>
                </div>
                <button
                  type="button"
                  onClick={sendMessage}
                  disabled={loading || !trimmedInput}
                  className="rounded-3xl bg-gradient-to-r from-amber-400 to-orange-400 px-6 py-4 text-sm font-semibold text-slate-950 transition hover:brightness-105 disabled:cursor-not-allowed disabled:opacity-70"
                >
                  {loading ? "Sending..." : "Send"}
                </button>
              </div>
            </div>
          </section>
        </div>
      </main>
    </>
  );
}
