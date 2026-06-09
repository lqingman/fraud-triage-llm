# RAG (optional — Phase 6)

**Not part of the core pipeline and not needed for it.** The fraud signal lives
in the transcript, and the LLM is fine-tuned to recognize it — there is no
external knowledge to retrieve for the base task.

Add this ONLY after Phases 0–5 work, as a "richer explanations" extension:
retrieve similar **known scam scripts / fraud-typology entries** and pass them
as context so the model can ground its `reason` in a known pattern
("matches a known SSN-refund scam playbook").

Slots in here without touching the trained model:
  - build a small embedded index of scam-typology docs (FAISS / Chroma)
  - retrieve top-k, inject into the prompt before generation in src/serve/app.py
