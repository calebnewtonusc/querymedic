"use client";

import Nav from "@/components/nav";
import Waitlist from "@/components/waitlist";

const ACCENT = "#14B8A6";
const HUB_URL = "https://specialized-model-startups.vercel.app";


function SectionLabel({ label }: { label: string }) {
  return (
    <div className="reveal flex items-center gap-5 mb-12">
      <span className="text-xs font-semibold uppercase tracking-[0.18em] text-gray-400 shrink-0">{label}</span>
      <div className="flex-1 h-px bg-gray-100" />
    </div>
  );
}

export default function Home() {
  return (
    <div className="min-h-screen bg-white text-[#0a0a0a] overflow-x-hidden">
      <Nav />

      {/* Hero */}
      <section className="relative min-h-screen flex flex-col justify-center px-6 pt-14 overflow-hidden">
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            backgroundImage: `radial-gradient(circle at 20% 30%, ${ACCENT}07 0%, transparent 50%), radial-gradient(circle at 80% 70%, ${ACCENT}05 0%, transparent 50%)`,
          }}
        />

        <div className="relative max-w-5xl mx-auto w-full py-20">
          <div className="fade-up delay-0 mb-8">
            <span
              className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-xs font-semibold border"
              style={{ color: ACCENT, borderColor: `${ACCENT}30`, backgroundColor: `${ACCENT}08` }}
            >
              <span className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ backgroundColor: ACCENT }} />
              Training &middot; Database Optimization &middot; ETA Q1 2027
            </span>
          </div>

          <h1 className="fade-up delay-1 text-[clamp(3rem,9vw,6.5rem)] font-bold leading-[0.92] tracking-tight mb-6">
            <span className="serif font-light italic" style={{ color: ACCENT }}>Query</span>
            <span>Medic</span>
          </h1>

          <p className="fade-up delay-2 serif text-[clamp(1.25rem,3vw,2rem)] font-light text-gray-500 mb-4 max-w-xl">
            Diagnose. Prescribe. Prove faster.
          </p>

          <p className="fade-up delay-3 text-sm text-gray-400 leading-relaxed max-w-lg mb-10">
            First model trained on EXPLAIN ANALYZE output &rarr; optimization pairs&nbsp;&mdash; understands PostgreSQL planner statistics, index types, and MVCC behavior, not just SQL syntax.
          </p>

          <div className="fade-up delay-4">
            <Waitlist />
          </div>
        </div>
      </section>

      {/* The Problem */}
      <section className="px-6 py-24 max-w-5xl mx-auto">
        <SectionLabel label="The Problem" />
        <div className="grid md:grid-cols-2 gap-6">
          <div className="reveal rounded-2xl border border-gray-100 p-8 bg-gray-50/50">
            <p className="text-xs font-semibold uppercase tracking-widest text-gray-400 mb-5">What general models do</p>
            <ul className="space-y-3 text-sm text-gray-500 leading-relaxed">
              <li className="flex gap-3">
                <span className="text-gray-300 mt-0.5">&#8212;</span>
                ORM tools generate schemas. Query analyzers show slow queries. Neither understands WHY.
              </li>
              <li className="flex gap-3">
                <span className="text-gray-300 mt-0.5">&#8212;</span>
                Can&apos;t explain why a specific join on 10M rows needs a specific index type to avoid a sequential scan
              </li>
              <li className="flex gap-3">
                <span className="text-gray-300 mt-0.5">&#8212;</span>
                Suggest &ldquo;add an index&rdquo; without understanding planner statistics, MVCC visibility, or write amplification
              </li>
              <li className="flex gap-3">
                <span className="text-gray-300 mt-0.5">&#8212;</span>
                No understanding of GIN vs GiST vs partial vs covering index tradeoffs per query pattern
              </li>
            </ul>
          </div>

          <div
            className="reveal rounded-2xl border p-8"
            style={{ borderColor: `${ACCENT}25`, backgroundColor: `${ACCENT}05` }}
          >
            <p className="text-xs font-semibold uppercase tracking-widest mb-5" style={{ color: ACCENT }}>What QueryMedic does</p>
            <ul className="space-y-3 text-sm leading-relaxed text-gray-700">
              <li className="flex gap-3">
                <svg className="mt-0.5 shrink-0" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={ACCENT} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
                Reads your EXPLAIN ANALYZE output the way a senior DBA does&nbsp;&mdash; planner estimates, actual rows, cost
              </li>
              <li className="flex gap-3">
                <svg className="mt-0.5 shrink-0" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={ACCENT} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
                Diagnoses the planner&apos;s misestimate and prescribes the exact index type or rewrite that fixes it
              </li>
              <li className="flex gap-3">
                <svg className="mt-0.5 shrink-0" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={ACCENT} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
                Comes with timing proof&nbsp;&mdash; before/after execution time, not theoretical improvement
              </li>
              <li className="flex gap-3">
                <svg className="mt-0.5 shrink-0" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={ACCENT} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
                Manages write amplification budget&nbsp;&mdash; faster reads without killing write throughput
              </li>
            </ul>
          </div>
        </div>
      </section>

      {/* How It&apos;s Built */}
      <section className="px-6 py-24 bg-gray-50/50">
        <div className="max-w-5xl mx-auto">
          <SectionLabel label="How it&apos;s built" />
          <div className="grid md:grid-cols-3 gap-6">
            {[
              {
                step: "01",
                title: "Supervised Fine-Tuning",
                desc: "300k EXPLAIN ANALYZE output + query rewrite + before/after timing triples from DBA Stack Exchange and engineering blogs. Each sample includes the full query plan, the optimization applied, and the measured execution time delta. QueryMedic learns to read query plans the way a senior DBA does.",
              },
              {
                step: "02",
                title: "RL with Verifiable Reward",
                desc: "Triple reward: query execution time improvement + index efficiency score (index scans vs sequential scans) + no write throughput regression. All three are automatically measurable from database benchmarks. QueryMedic is penalized for index designs that improve read latency but cause write amplification spikes.",
              },
              {
                step: "03",
                title: "DPO Alignment",
                desc: "Direct Preference Optimization on (targeted index prescription, generic index suggestion) pairs. QueryMedic learns to prefer partial indexes over full-table indexes when selectivity warrants it, and covering indexes over index + heap fetch when the query hot path requires it.",
              },
            ].map(({ step, title, desc }) => {
              return (
                <div key={step} className="reveal-scale rounded-2xl border border-gray-100 bg-white p-8">
                  <div className="text-xs font-bold uppercase tracking-widest mb-4" style={{ color: ACCENT }}>{step}</div>
                  <h3 className="serif font-semibold text-lg mb-3 text-gray-900">{title}</h3>
                  <p className="text-sm text-gray-500 leading-relaxed">{desc}</p>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Capabilities */}
      <section className="px-6 py-24 max-w-5xl mx-auto">
        <SectionLabel label="Capabilities" />
        <div className="grid sm:grid-cols-2 gap-5">
          {[
            {
              icon: (
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke={ACCENT} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>
                </svg>
              ),
              title: "EXPLAIN ANALYZE interpretation — senior DBA level plan reading",
              desc: "Reads query plans across PostgreSQL, MySQL, and SQLite. Identifies planner misestimates, nested loop vs hash join vs merge join selection errors, and row count estimation failures from stale statistics.",
            },
            {
              icon: (
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke={ACCENT} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>
                </svg>
              ),
              title: "Workload-aware index design per query pattern",
              desc: "Selects GIN vs GiST vs B-tree vs partial vs covering index based on query selectivity, update frequency, and MVCC dead tuple accumulation. Explains why each index type was chosen for the specific access pattern.",
            },
            {
              icon: (
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke={ACCENT} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/>
                </svg>
              ),
              title: "Query rewrite for better planner selectivity estimates",
              desc: "Rewrites queries to expose selectivity information the planner can use&nbsp;&mdash; flattening subqueries, reordering joins to match statistics, and adding explicit filter pushdowns to improve row count estimates.",
            },
            {
              icon: (
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke={ACCENT} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="12" y1="20" x2="12" y2="10"/><line x1="18" y1="20" x2="18" y2="4"/><line x1="6" y1="20" x2="6" y2="16"/>
                </svg>
              ),
              title: "Write amplification budget management",
              desc: "Tracks the write amplification cost of every index recommendation. Manages total index maintenance overhead against a configurable write throughput budget&nbsp;&mdash; faster reads without degrading INSERT/UPDATE performance.",
            },
          ].map(({ icon, title, desc }) => {
            return (
              <div
                key={title}
               
                className="reveal rounded-2xl border border-gray-100 p-7 flex gap-5 hover:border-gray-200 transition-colors"
              >
                <div
                  className="shrink-0 w-10 h-10 rounded-xl flex items-center justify-center"
                  style={{ backgroundColor: `${ACCENT}10` }}
                >
                  {icon}
                </div>
                <div>
                  <h3 className="font-semibold text-sm text-gray-900 mb-1.5">{title}</h3>
                  <p className="text-sm text-gray-500 leading-relaxed">{desc}</p>
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* The Numbers */}
      <section className="px-6 py-24 bg-gray-50/50">
        <div className="max-w-5xl mx-auto">
          <SectionLabel label="The numbers" />
          <div className="grid sm:grid-cols-3 gap-6">
            {[
              { stat: "300k", label: "Training triples", sub: "EXPLAIN ANALYZE + rewrite + timing from DBA Stack Exchange + blogs" },
              { stat: "Qwen2.5-7B", label: "Base model", sub: "Coder-Instruct" },
              { stat: "3-part", label: "Reward signal", sub: "Execution time + index efficiency + write throughput" },
            ].map(({ stat, label, sub }) => {
              return (
                <div
                  key={label}
                 
                  className="reveal rounded-2xl border p-8"
                  style={{ borderColor: `${ACCENT}20` }}
                >
                  <div className="text-3xl font-bold tracking-tight mb-2" style={{ color: ACCENT }}>{stat}</div>
                  <div className="text-sm font-semibold text-gray-800 mb-1">{label}</div>
                  <div className="text-xs text-gray-400">{sub}</div>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="px-6 py-12 border-t border-gray-100">
        <div className="max-w-5xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4 text-sm text-gray-400">
          <p>
            Part of the{" "}
            <a href={HUB_URL} className="underline underline-offset-2 hover:text-gray-600 transition-colors">
              Specialist AI
            </a>{" "}
            portfolio &middot; Caleb Newton &middot; USC &middot; 2026
          </p>
        </div>
      </footer>
    </div>
  );
}
