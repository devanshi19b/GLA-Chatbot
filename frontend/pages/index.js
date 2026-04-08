import Head from "next/head";
import { useEffect, useRef, useState } from "react";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

const starterMessage = {
  role: "assistant",
  content:
    "Ask me anything about GLA University. I answer from the official GLA website first and from other indexed sources only when needed.",
  sources: [],
};

const exampleQuestions = [
  "What programmes are mentioned on the official website?",
  "What does GLA say about placements?",
  "How many labs and faculty members are mentioned?",
];

async function readResponsePayload(response) {
  const contentType = response.headers.get("content-type") || "";

  if (contentType.includes("application/json")) {
    try {
      return await response.json();
    } catch {
      return null;
    }
  }

  try {
    return await response.text();
  } catch {
    return null;
  }
}

export default function Home() {
  const [messages, setMessages] = useState([starterMessage]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [showTools, setShowTools] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [ingestLoading, setIngestLoading] = useState(false);
  const [ingestError, setIngestError] = useState("");
  const [ingestStatus, setIngestStatus] = useState("");
  const [crawlUrl, setCrawlUrl] = useState("https://www.gla.ac.in/info/common/");
  const [crawlPages, setCrawlPages] = useState(12);
  const bottomRef = useRef(null);
  const fileInputRef = useRef(null);
  const trimmedInput = input.trim();
  const visibleMessages = messages.slice(1);
  const showWelcome = visibleMessages.length === 0;

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const appendAssistantError = (detail) => {
    setError(detail);
    setMessages((currentMessages) => [
      ...currentMessages,
      {
        role: "assistant",
        content:
          "I couldn't fetch a grounded answer because the backend request failed. Please verify the API server and source sync, then try again.",
        sources: [],
      },
    ]);
  };

  const sendMessage = async () => {
    const nextInput = input.trim();
    if (!nextInput || loading) {
      return;
    }

    setError("");
    setLoading(true);
    setInput("");

    const userMessage = { role: "user", content: nextInput };
    setMessages((currentMessages) => [...currentMessages, userMessage]);

    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: nextInput,
        }),
      });
      const payload = await readResponsePayload(response);

      if (!response.ok) {
        throw new Error(payload?.detail || "Chat request failed.");
      }

      setMessages((currentMessages) => [
        ...currentMessages,
        {
          role: "assistant",
          content: payload?.answer,
          sources: payload?.sources || [],
        },
      ]);
    } catch (requestError) {
      const detail =
        requestError instanceof Error
          ? requestError.message
          : "The chatbot could not reach the backend. Please check that the FastAPI server is running.";
      appendAssistantError(detail);
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

  const handleFileSelection = (event) => {
    setSelectedFiles(Array.from(event.target.files || []));
    setIngestError("");
    setIngestStatus("");
  };

  const uploadSelectedFiles = async () => {
    if (!selectedFiles.length || ingestLoading) {
      return;
    }

    setIngestLoading(true);
    setIngestError("");
    setIngestStatus("");

    const formData = new FormData();
    selectedFiles.forEach((file) => {
      formData.append("files", file);
    });

    try {
      const response = await fetch(`${API_BASE_URL}/ingest/upload`, {
        method: "POST",
        body: formData,
      });
      const payload = await readResponsePayload(response);

      if (!response.ok) {
        throw new Error(payload?.detail || "Upload failed. Please try again.");
      }

      setIngestStatus(payload?.message || "Files uploaded and indexed.");
      setSelectedFiles([]);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    } catch (requestError) {
      const detail =
        requestError instanceof Error ? requestError.message : "Upload failed. Please try again.";
      setIngestError(detail);
    } finally {
      setIngestLoading(false);
    }
  };

  const crawlWebsite = async () => {
    if (!crawlUrl.trim() || ingestLoading) {
      return;
    }

    setIngestLoading(true);
    setIngestError("");
    setIngestStatus("");

    try {
      const response = await fetch(`${API_BASE_URL}/ingest/web`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          url: crawlUrl.trim(),
          max_pages: Number(crawlPages) || 12,
        }),
      });
      const payload = await readResponsePayload(response);

      if (!response.ok) {
        throw new Error(payload?.detail || "Website crawl failed. Please try again.");
      }

      setIngestStatus(payload?.message || "Website pages were crawled and indexed successfully.");
    } catch (requestError) {
      const detail =
        requestError instanceof Error
          ? requestError.message
          : "Website crawl failed. Please try again.";
      setIngestError(detail);
    } finally {
      setIngestLoading(false);
    }
  };

  const syncOfficialWebsite = async () => {
    if (ingestLoading) {
      return;
    }

    setIngestLoading(true);
    setIngestError("");
    setIngestStatus("");

    try {
      const response = await fetch(`${API_BASE_URL}/ingest/official`, {
        method: "POST",
      });
      const payload = await readResponsePayload(response);

      if (!response.ok) {
        throw new Error(payload?.detail || "Official website sync failed.");
      }

      setIngestStatus(payload?.message || "Official GLA website synced successfully.");
    } catch (requestError) {
      const detail =
        requestError instanceof Error
          ? requestError.message
          : "Official website sync failed.";
      setIngestError(detail);
    } finally {
      setIngestLoading(false);
    }
  };

  return (
    <>
      <Head>
        <title>GLA University Assistant</title>
        <meta
          name="description"
          content="A clean, classic and vibrant GLA University chatbot connected directly to official website content and indexed sources."
        />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      <main className="h-screen overflow-hidden px-3 py-3 sm:px-5 sm:py-5">
        <div className="relative mx-auto flex h-full max-w-[1500px] overflow-hidden rounded-[36px] border border-white/20 bg-[linear-gradient(135deg,#8c2dff_0%,#6b63ff_38%,#4b8cff_100%)] shadow-[0_40px_120px_rgba(62,32,161,0.38)]">
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_40%,rgba(255,255,255,0.16),transparent_24%),radial-gradient(circle_at_18%_16%,rgba(255,255,255,0.08),transparent_20%),radial-gradient(circle_at_85%_88%,rgba(255,183,77,0.16),transparent_22%)]" />

          <div className="relative flex min-w-0 flex-1 flex-col items-center justify-center px-4 py-4 sm:px-8">
            <div className="flex w-full max-w-[860px] flex-col overflow-hidden rounded-[34px] border border-white/30 bg-[linear-gradient(180deg,rgba(255,255,255,0.68),rgba(255,255,255,0.9))] shadow-[0_28px_80px_rgba(34,18,86,0.28)] backdrop-blur-2xl">
              <header className="flex items-center justify-between gap-3 border-b border-white/40 bg-[linear-gradient(90deg,rgba(150,112,255,0.9),rgba(115,155,255,0.9))] px-5 py-4 text-white sm:px-6">
                <div className="flex items-center gap-3">
                  <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-white/18 text-sm font-semibold shadow-[0_10px_24px_rgba(255,255,255,0.18)]">
                    GLA
                  </div>
                  <div>
                    <p className="text-base font-semibold sm:text-lg">Talk to GLA Assistant</p>
                    <p className="text-xs text-white/80">
                      Official website first, indexed sources only
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => setShowTools((current) => !current)}
                    className="rounded-full border border-white/25 bg-white/12 px-4 py-2 text-xs font-semibold uppercase tracking-[0.16em] text-white transition hover:bg-white/20"
                  >
                    {showTools ? "Close" : "Data"}
                  </button>
                  <button
                    type="button"
                    onClick={clearChat}
                    disabled={loading}
                    className="rounded-full bg-white px-4 py-2 text-xs font-semibold uppercase tracking-[0.16em] text-[#5b40f0] transition hover:brightness-95 disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    Clear
                  </button>
                </div>
              </header>

              <div className="relative flex min-h-0 flex-1 flex-col bg-[radial-gradient(circle_at_top,rgba(255,255,255,0.75),rgba(252,250,255,0.92)_42%,rgba(246,247,255,0.98)_100%)]">
                <div className="message-scrollbar flex-1 overflow-y-auto px-4 py-5 sm:px-8 sm:py-8">
                  {showWelcome ? (
                    <div className="flex min-h-[420px] flex-col items-center justify-center text-center">
                      <div className="flex h-24 w-24 items-center justify-center rounded-full bg-[radial-gradient(circle_at_35%_35%,#ffffff,#f3ddff_34%,#c89fff_58%,#8e66ff_78%,#5d7eff_100%)] shadow-[0_20px_50px_rgba(118,102,255,0.32)]" />
                      <p className="mt-8 text-3xl font-semibold leading-tight text-[#3c2c79] sm:text-4xl">
                        Welcome to GLA University.
                        <br />
                        <span className="text-[#6174ff]">How may I help you today?</span>
                      </p>
                      <p className="mt-4 max-w-2xl text-sm leading-7 text-[#6f6895] sm:text-base">
                        A clean GLA knowledge desk built for fast university queries. Sync the
                        official website once, then ask naturally.
                      </p>

                      <div className="mt-8 flex flex-wrap justify-center gap-3">
                        {exampleQuestions.map((question) => (
                          <button
                            key={question}
                            type="button"
                            onClick={() => handleExampleClick(question)}
                            className="rounded-full border border-[#d9d4ff] bg-white/85 px-4 py-2 text-sm text-[#5e5789] transition hover:border-[#8d86ff] hover:text-[#362f63]"
                          >
                            {question}
                          </button>
                        ))}
                      </div>
                    </div>
                  ) : (
                    <div className="mx-auto flex w-full max-w-3xl flex-col gap-4">
                      <div className="self-center rounded-full bg-white/70 px-4 py-1 text-xs font-medium text-[#8e84ba] shadow-sm">
                        Today
                      </div>

                      {visibleMessages.map((message, index) => (
                        <div
                          key={`${message.role}-${index}`}
                          className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
                        >
                          <div
                            className={`max-w-[82%] rounded-[24px] px-4 py-3 text-sm leading-7 shadow-[0_18px_40px_rgba(39,24,105,0.08)] sm:px-5 ${
                              message.role === "user"
                                ? "rounded-br-md bg-[linear-gradient(90deg,#8e6dff,#6c86ff)] text-white"
                                : "rounded-bl-md border border-white/70 bg-white/88 text-[#5d5782]"
                            }`}
                          >
                            <p className="whitespace-pre-wrap">{message.content}</p>
                          </div>
                        </div>
                      ))}

                      {loading && (
                        <div className="flex justify-start">
                          <div className="max-w-sm rounded-[24px] rounded-bl-md border border-white/70 bg-white/88 px-4 py-3 text-sm text-[#7b739f] shadow-[0_18px_40px_rgba(39,24,105,0.08)]">
                            Searching the indexed GLA sources...
                          </div>
                        </div>
                      )}

                      <div ref={bottomRef} />
                    </div>
                  )}
                </div>

                <div className="border-t border-white/60 bg-white/55 px-4 py-4 backdrop-blur-md sm:px-6">
                  {error && (
                    <p className="mx-auto mb-3 max-w-3xl rounded-2xl border border-red-300/40 bg-red-50/90 px-4 py-3 text-sm text-red-600">
                      {error}
                    </p>
                  )}

                  <div className="mx-auto max-w-3xl rounded-[28px] border border-[#f0d8ff] bg-white p-3 shadow-[0_24px_70px_rgba(74,38,176,0.12)]">
                    <textarea
                      value={input}
                      onChange={(event) => setInput(event.target.value)}
                      onKeyDown={handleKeyDown}
                      rows={3}
                      placeholder="Type your question here..."
                      className="min-h-[92px] w-full resize-none rounded-[22px] border border-[#f3e7ff] bg-[#fcfbff] px-4 py-3 text-sm text-[#5c5880] outline-none transition focus:border-[#a698ff] focus:ring-2 focus:ring-[#ddd7ff]"
                    />

                    <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
                      <div className="flex flex-wrap gap-2">
                        <button
                          type="button"
                          onClick={syncOfficialWebsite}
                          disabled={ingestLoading}
                          className="rounded-full border border-[#ecd8ff] bg-[#fff7ff] px-3 py-2 text-xs font-medium text-[#7d59d8] transition hover:border-[#b69bff] disabled:cursor-not-allowed disabled:opacity-60"
                        >
                          {ingestLoading ? "Syncing..." : "Sync official website"}
                        </button>
                        <button
                          type="button"
                          onClick={() => handleExampleClick(exampleQuestions[0])}
                          className="rounded-full border border-slate-200 bg-white px-3 py-2 text-xs text-slate-500 transition hover:border-[#8d86ff] hover:text-[#4b4378]"
                        >
                          Suggested prompt
                        </button>
                      </div>

                      <div className="flex items-center gap-3">
                        <button
                          type="button"
                          onClick={sendMessage}
                          disabled={loading || !trimmedInput}
                          className="rounded-2xl bg-[linear-gradient(90deg,#a44cff,#6c86ff)] px-5 py-3 text-sm font-semibold text-white transition hover:brightness-105 disabled:cursor-not-allowed disabled:opacity-70"
                        >
                          {loading ? "Sending..." : "Send"}
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {showTools && (
              <aside className="message-scrollbar absolute inset-y-5 right-5 z-10 w-[340px] overflow-y-auto rounded-[30px] border border-white/30 bg-white/92 p-4 shadow-[0_30px_90px_rgba(37,18,117,0.22)] backdrop-blur-xl">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-[0.22em] text-[#9a8fd1]">
                      Source control
                    </p>
                    <p className="mt-2 text-sm leading-6 text-[#6e6794]">
                      Use the official website as the main source, then add extra files only when
                      needed.
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={() => setShowTools(false)}
                    className="rounded-full border border-slate-200 px-3 py-1 text-xs font-semibold text-slate-500 transition hover:text-slate-900"
                  >
                    Close
                  </button>
                </div>

                {ingestError && (
                  <p className="mt-4 rounded-2xl border border-red-300/50 bg-red-50 px-4 py-3 text-sm text-red-600">
                    {ingestError}
                  </p>
                )}

                {ingestStatus && (
                  <p className="mt-4 rounded-2xl border border-emerald-300/50 bg-emerald-50 px-4 py-3 text-sm text-emerald-700">
                    {ingestStatus}
                  </p>
                )}

                <div className="mt-4 rounded-[26px] border border-[#efe4ff] bg-[linear-gradient(180deg,#fff8ff,#f8f7ff)] p-4">
                  <p className="text-xs font-semibold uppercase tracking-[0.2em] text-[#9a8fd1]">
                    Official website
                  </p>
                  <p className="mt-2 text-sm leading-6 text-[#6e6794]">
                    Pull directly from the official GLA website so you do not have to keep every
                    detail in PDFs.
                  </p>
                  <button
                    type="button"
                    onClick={syncOfficialWebsite}
                    disabled={ingestLoading}
                    className="mt-4 w-full rounded-full bg-[linear-gradient(90deg,#a44cff,#6c86ff)] px-4 py-3 text-sm font-semibold text-white transition hover:brightness-105 disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    {ingestLoading ? "Syncing..." : "Sync official GLA website"}
                  </button>
                </div>

                <div className="mt-4 rounded-[26px] border border-[#efe4ff] bg-[linear-gradient(180deg,#fff8ff,#f8f7ff)] p-4">
                  <p className="text-xs font-semibold uppercase tracking-[0.2em] text-[#9a8fd1]">
                    Custom crawl
                  </p>
                  <input
                    type="url"
                    value={crawlUrl}
                    onChange={(event) => setCrawlUrl(event.target.value)}
                    placeholder="https://www.gla.ac.in/info/common/"
                    className="mt-3 w-full rounded-2xl border border-[#eadfff] bg-white px-4 py-3 text-sm text-[#5e5789] outline-none transition focus:border-[#9a90ff] focus:ring-2 focus:ring-[#ddd7ff]"
                  />
                  <div className="mt-3 flex items-center gap-3">
                    <input
                      type="number"
                      min="1"
                      max="50"
                      value={crawlPages}
                      onChange={(event) => setCrawlPages(event.target.value)}
                      className="w-24 rounded-2xl border border-[#eadfff] bg-white px-4 py-3 text-sm text-[#5e5789] outline-none transition focus:border-[#9a90ff] focus:ring-2 focus:ring-[#ddd7ff]"
                    />
                    <p className="text-xs leading-5 text-[#948db6]">
                      Crawl any official GLA page you want indexed.
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={crawlWebsite}
                    disabled={ingestLoading || !crawlUrl.trim()}
                    className="mt-4 w-full rounded-full border border-[#c3b2ff] px-4 py-3 text-sm font-semibold text-[#7346ef] transition hover:bg-white disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    {ingestLoading ? "Indexing..." : "Crawl and index"}
                  </button>
                </div>

                <div className="mt-4 rounded-[26px] border border-[#efe4ff] bg-[linear-gradient(180deg,#fff8ff,#f8f7ff)] p-4">
                  <p className="text-xs font-semibold uppercase tracking-[0.2em] text-[#9a8fd1]">
                    Upload files
                  </p>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".pdf,.txt,.md"
                    multiple
                    onChange={handleFileSelection}
                    className="mt-3 block w-full text-sm text-[#6e6794] file:mr-3 file:rounded-full file:border-0 file:bg-[#8f98ff] file:px-4 file:py-2 file:text-sm file:font-semibold file:text-white hover:file:brightness-105"
                  />
                  {selectedFiles.length > 0 && (
                    <p className="mt-2 text-xs leading-5 text-[#948db6]">{selectedFiles.length} file(s) selected</p>
                  )}
                  <button
                    type="button"
                    onClick={uploadSelectedFiles}
                    disabled={ingestLoading || selectedFiles.length === 0}
                    className="mt-4 w-full rounded-full border border-[#c3b2ff] px-4 py-3 text-sm font-semibold text-[#7346ef] transition hover:bg-white disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    {ingestLoading ? "Indexing..." : "Upload and index"}
                  </button>
                </div>
              </aside>
            )}
          </div>
        </div>
      </main>
    </>
  );
}
