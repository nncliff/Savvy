import { MessageCircle } from "lucide-react";
import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";

export default function RightRailChat() {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState([
    { role: "system", content: "Ask me anything about your bookmarks!" },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, open]);

  async function handleSend() {
    if (!input.trim()) return;
    setLoading(true);
    setMessages((msgs) => [...msgs, { role: "user", content: input }]);
    try {
      const res = await fetch("/api/llamaindex", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: input }),
      });
      let data = null;
      try {
        data = await res.json();
      } catch (jsonErr) {
        // Ignore JSON parse errors
      }
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      setMessages((msgs) => [
        ...msgs,
        { role: "assistant", content: data?.response || data?.error || "No response." },
      ]);
    } catch (e) {
      setMessages((msgs) => [
        ...msgs,
        { role: "assistant", content: `Error contacting server: ${e instanceof Error ? e.message : e}` },
      ]);
    }
    setInput("");
    setLoading(false);
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter" && !loading) handleSend();
  }

  return (
    <>
      <button
        className="fixed bottom-20 right-6 z-[2147483647] bg-blue-600 text-white rounded-full p-4 shadow-lg hover:bg-blue-700 focus:outline-none"
        style={{ boxShadow: '0 4px 24px rgba(0,0,0,0.18)', right: '1.5rem', bottom: '5rem', left: 'auto', top: 'auto', zIndex: 2147483647 }}
        onClick={() => setOpen((v) => !v)}
        aria-label="Open chat"
      >
        <MessageCircle size={28} />
      </button>
      {open && (
        <aside className="fixed right-0 top-0 z-[1100] h-full w-full max-w-md border-l bg-white shadow-lg flex flex-col animate-slideIn">
          <div className="flex justify-between items-center p-4 border-b">
            <span className="font-semibold">Bookmark Chat</span>
            <button
              className="text-gray-500 hover:text-gray-800"
              onClick={() => setOpen(false)}
              aria-label="Close chat"
            >
              ×
            </button>
          </div>
          <div className="flex-1 overflow-y-auto p-4 space-y-3">
            {messages.map((msg, i) => (
              <div
                key={i}
                className={
                  msg.role === "user"
                    ? "flex justify-end"
                    : msg.role === "assistant"
                    ? "flex justify-start"
                    : "flex justify-center"
                }
              >
                {msg.role === "system" ? (
                  <div className="text-center text-gray-500 text-xs bg-gray-100 px-3 py-1 rounded-full">
                    {msg.content}
                  </div>
                ) : (
                  <div
                    className={
                      msg.role === "user"
                        ? "max-w-[80%] bg-blue-500 text-gray-900 px-4 py-2 rounded-2xl rounded-br-md shadow-md border border-blue-600"
                        : "max-w-[85%] bg-gray-50 text-gray-900 px-4 py-3 rounded-2xl rounded-bl-md shadow-md border border-gray-200"
                    }
                  >
                    {msg.role === "assistant" ? (
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        components={{
                          code({ inline, className, children, ...props }: any) {
                            const match = /language-(\w+)/.exec(className || "");
                            return !inline && match ? (
                              <SyntaxHighlighter
                                style={oneDark as any}
                                language={match[1]}
                                PreTag="div"
                                className="rounded-md text-sm my-2"
                                {...props}
                              >
                                {String(children).replace(/\n$/, "")}
                              </SyntaxHighlighter>
                            ) : (
                              <code
                                className="bg-gray-200 text-gray-800 px-1.5 py-0.5 rounded text-sm font-mono"
                                {...props}
                              >
                                {children}
                              </code>
                            );
                          },
                          p: ({ children }) => (
                            <p className="mb-2 last:mb-0 leading-relaxed">{children}</p>
                          ),
                          ul: ({ children }) => (
                            <ul className="list-disc list-inside mb-2 space-y-1 ml-2">{children}</ul>
                          ),
                          ol: ({ children }) => (
                            <ol className="list-decimal list-inside mb-2 space-y-1 ml-2">{children}</ol>
                          ),
                          h1: ({ children }) => (
                            <h1 className="text-lg font-bold mb-2 mt-1">{children}</h1>
                          ),
                          h2: ({ children }) => (
                            <h2 className="text-base font-bold mb-2 mt-1">{children}</h2>
                          ),
                          h3: ({ children }) => (
                            <h3 className="text-sm font-bold mb-1 mt-1">{children}</h3>
                          ),
                          blockquote: ({ children }) => (
                            <blockquote className="border-l-4 border-gray-400 pl-4 italic mb-2 ml-2">
                              {children}
                            </blockquote>
                          ),
                          strong: ({ children }) => (
                            <strong className="font-semibold">{children}</strong>
                          ),
                          em: ({ children }) => (
                            <em className="italic">{children}</em>
                          ),
                          a: ({ href, children }) => (
                            <a
                              href={href}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-blue-600 hover:text-blue-700 underline font-medium"
                            >
                              {children}
                            </a>
                          ),
                        }}
                      >
                        {msg.content}
                      </ReactMarkdown>
                    ) : (
                      <div className="leading-relaxed">{msg.content}</div>
                    )}
                  </div>
                )}
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
          <div className="p-4 border-t flex gap-2">
            <input
              className="flex-1 border rounded px-3 py-2"
              type="text"
              placeholder="Type your question..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={loading}
            />
            <button
              className="bg-blue-600 text-white px-4 py-2 rounded disabled:opacity-50"
              onClick={handleSend}
              disabled={loading}
            >
              Send
            </button>
          </div>
        </aside>
      )}
      <style>{`
        @keyframes slideIn {
          from { transform: translateX(100%); }
          to { transform: translateX(0); }
        }
        .animate-slideIn {
          animation: slideIn 0.2s cubic-bezier(0.4,0,0.2,1);
        }
      `}</style>
    </>
  );
}
