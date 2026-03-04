"use client";

import { useState } from "react";

const ACCENT = "#14B8A6";

export default function Waitlist() {
  const [email, setEmail] = useState("");
  const [submitted, setSubmitted] = useState(false);

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (email.trim()) {
      setSubmitted(true);
    }
  }

  if (submitted) {
    return (
      <div className="flex items-center gap-2 text-sm font-medium" style={{ color: ACCENT }}>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
          <polyline points="20 6 9 17 4 12"/>
        </svg>
        You&apos;re on the list. We&apos;ll be in touch.
      </div>
    );
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col sm:flex-row gap-3 max-w-md">
      <input
        type="email"
        required
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        placeholder="you@example.com"
        className="flex-1 px-4 py-2.5 text-sm border border-gray-200 rounded-lg outline-none transition-colors bg-white text-gray-900 placeholder:text-gray-400"
        style={{ outlineColor: ACCENT }}
      />
      <button
        type="submit"
        className="px-5 py-2.5 text-sm font-semibold text-white rounded-lg transition-opacity hover:opacity-90"
        style={{ backgroundColor: ACCENT }}
      >
        Join waitlist
      </button>
    </form>
  );
}
