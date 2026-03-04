"use client";
import { useEffect } from "react";

export default function RevealObserver() {
  useEffect(() => {
    const obs = new IntersectionObserver(
      (entries) =>
        entries.forEach((e) => {
          if (e.isIntersecting) {
            e.target.classList.add("visible");
            obs.unobserve(e.target);
          }
        }),
      { threshold: 0 }
    );

    function processAll() {
      document.querySelectorAll(".reveal, .reveal-scale").forEach((el) => {
        if (el.classList.contains("visible")) return;
        const rect = el.getBoundingClientRect();
        const inView = rect.top < window.innerHeight && rect.bottom > 0;
        if (inView) {
          el.classList.add("visible");
        } else {
          obs.observe(el);
        }
      });
    }

    processAll();
    const timers = [50, 150, 300, 600].map((d) => setTimeout(processAll, d));

    const mut = new MutationObserver(processAll);
    mut.observe(document.body, { childList: true, subtree: true });

    return () => {
      obs.disconnect();
      mut.disconnect();
      timers.forEach(clearTimeout);
    };
  }, []);
  return null;
}
