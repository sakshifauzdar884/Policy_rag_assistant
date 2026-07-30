Policy RAG Assistant

Oerview:An AI-powered assistant that answers employee questions about **expense,travel, finance, and HR policy** using Retrieval-Augmented Generation (RAGIt answers strictly from the provided documents, always cites its sources,supports multi-turn follow-up questions, and clearly says so when the answerisn't in the documents rather than guessing.

Built for the "AI Engineer (RAG)" assignment.

1. Important assumption about the input documents
The assignment named four documents (Expense Policy, Travel Policy, Finance Policy, Employee Handbook) but did not attach their actual content. To deliver a fully working, testable system, I authored **realistic sample versions** of all four policies with plausible sections, numbers, and edge cases (reimbursement limits, approval thresholds, leave entitlements, etc.).

2. Setup instructions

```bash
# 1. Clone / unzip the project, then from the project root:
pip install -r requirements.txt

# 2. (Optional) Enable LLM-generated prose answers instead of the
#    extractive fallback (see "LLM backend" below):
export ANTHROPIC_API_KEY=sk-ant-...

# 3. Run the automated tests
python -m pytest tests/ -v          # or: python tests/test_rag.py

# 4. Run the scripted demo (prints sample Q&A with citations)
python run_demo.py

# 5. Chat interactively
python -m src.cli
python -m src.cli --rebuild                     # force a fresh index build
python -m src.cli --filter expense_policy       # restrict search to one document
```

The first run builds the index from `data/` and caches it to
`index_store/index.pkl`; later runs load the cache unless `--rebuild` is
passed or `add_document()` (incremental indexing) is called.

**No API key is required to run anything above.** Without
`ANTHROPIC_API_KEY`, the generator falls back to a deterministic mode that
returns the top-matching passages directly with citations — the retrieval,
citation, refusal, and conversation logic are all still fully exercised.
With the key set, answers become fluent LLM-synthesized prose, still
constrained to only the retrieved passages and still cited.

---

## 3. Architecture overview

```
                 ┌────────────────────┐
   data/*.md ──▶ │  document_loader   │  reads files, tags doc_type metadata
                 └─────────┬──────────┘
                           ▼
                 ┌────────────────────┐
                 │      chunker       │  splits on Markdown headers, then
                 │                    │  word-windows long sections (180w,
                 │                    │  30w overlap); keeps a section
                 │                    │  breadcrumb per chunk for citations
                 └─────────┬──────────┘
                           ▼
                 ┌────────────────────┐
                 │      indexer       │  builds TF-IDF matrix (sklearn) +
                 │   (RagIndex)       │  BM25 (custom impl) over the same
                 │                    │  chunk store; persists to disk
                 └─────────┬──────────┘
                           ▼
User question ─▶ ┌────────────────────┐
                 │  generator:        │  rewrites follow-up questions into
                 │  query rewrite     │  standalone queries using history
                 └─────────┬──────────┘
                           ▼
                 ┌────────────────────┐
                 │    retriever       │  query expansion → hybrid
                 │ (HybridRetriever)  │  TF-IDF + BM25 search → optional
                 │                    │  metadata filter → lexical rerank
                 └─────────┬──────────┘
                           ▼
                 ┌────────────────────┐
                 │     generator:     │  confidence gate ("not found" if
                 │  answer synthesis  │  score/overlap too low) → LLM (if
                 │                    │  configured) or extractive fallback
                 │                    │  → citations attached
                 └─────────┬──────────┘
                           ▼
                    Answer + citations
                 (rag_pipeline.RagPipeline
                  orchestrates all of the above,
                  exposed via src/cli.py)
```

**Module map:**
| Module | Responsibility |
|---|---|
| `src/document_loader.py` | Reads `data/`, infers `doc_type` from filename for metadata filtering |
| `src/chunker.py` | Header-aware chunking with a readable section breadcrumb per chunk |
| `src/bm25.py` | Dependency-free Okapi BM25 implementation + tokenizer/stopwords |
| `src/indexer.py` | Builds/persists the TF-IDF + BM25 index; incremental `add_documents` |
| `src/retriever.py` | Hybrid search, query expansion, metadata filtering, lexical rerank |
| `src/generator.py` | Confidence gating, citation formatting, LLM/extractive answer generation, follow-up query rewriting |
| `src/rag_pipeline.py` | Orchestrates the above behind a simple `.ask()` API with conversation state |
| `src/cli.py` | Interactive terminal chat |
| `run_demo.py` | Scripted sample questions (used to produce Section 4 below) |
| `tests/test_rag.py` | Automated smoke tests (indexing, citations, refusal, filtering, incremental add) |

---

## 4. Sample questions and outputs

Actual output from `python run_demo.py` (extractive mode, no API key set):

**Q: What is the daily meal reimbursement limit for domestic travel?**
> Domestic travel: INR 1,500 per day for incidentals (meals, local
> transport) in lieu of itemized receipts...
> **Citations:** Travel Policy § 6. Per Diem; Travel Policy § 1. Purpose; Travel Policy § 2. Trip Approval

**Q: What about international trips?** *(follow-up — no document named, relies on conversation history)*
> Rewritten for retrieval as: "What is the daily meal reimbursement limit
> for domestic travel? What about international trips?" → returns the
> international per-diem and travel-insurance passages.
> **Citations:** Travel Policy § 2. Trip Approval; Travel Policy § 6. Per Diem; Travel Policy § 7. Travel Insurance

**Q: How many days of annual leave do full-time employees get?**
> Full-time employees accrue 21 days of paid annual leave per calendar
> year, credited at 1.75 days per month...
> **Citations:** Employee Handbook § 3.1 Annual Leave; § 3.2 Sick Leave; § 2. Working Hours

**Q: What is the approval process for a purchase order above 25000 rupees?**
> Any purchase above INR 25,000 requires a Purchase Order (PO) raised in
> the Finance system before the vendor is engaged...
> **Citations:** Finance Policy § 3. Purchase Orders; § 6. Capital Expenditure (CapEx); § 2. Budget Ownership

**Q: Can I expense my gym membership?**
> The following are never reimbursable: parking or traffic fines, personal
> entertainment, **gym memberships**, clothing (except required safety
> gear)...
> **Citations:** Expense Policy § 4. Non-Reimbursable Expenses; § 5. Submission Process; § 7. Reimbursement Timeline

**Q: What is the company's stock price target for next year?** *(deliberately out of scope)*
> I could not find sufficient information in the provided documents
> (Expense Policy, Travel Policy, Finance Policy, Employee Handbook) to
> answer this question confidently. Please rephrase, or check with
> HR/Finance directly.

The full raw log is reproducible with `python run_demo.py`.

---

## 5. Design decisions

- **Header-aware chunking over fixed-size chunking.** Splitting on `##`/`###`
  headers first (falling back to word-windows only for long sections) keeps
  each chunk aligned to one policy topic, so citations read as
  "Travel Policy › 6. Per Diem" instead of an arbitrary offset — much more
  useful to an employee who wants to verify the source.

- **Hybrid search (TF-IDF cosine + BM25) instead of dense embeddings.**
  A downloaded sentence-embedding model wasn't guaranteed to be available in
  every deployment/grading environment (no network access, in this sandbox).
  TF-IDF captures topical/semantic-ish overlap; BM25 (implemented from
  scratch in ~60 lines, `src/bm25.py`) is a strong keyword signal for short,
  specific queries ("25,000 rupees", "gym membership"). Combining both is
  the assignment's **Hybrid Search** bonus, achieved without any model
  download — swapping in real embeddings later (see Section 7) is a drop-in
  change to `retriever.py`.

- **Metadata filtering** is auto-inferred from the query (a small
  keyword→doc_type table in `retriever.py`) but only applied when exactly
  one document type matches strongly; ambiguous queries search everything
  rather than risk wrongly excluding the right document. Can also be set
  explicitly (`--filter travel_policy` on the CLI).

- **Query expansion** uses a small hand-authored synonym table (e.g.
  "cab" → "taxi, auto, ride") appended to the query before retrieval. Kept
  intentionally simple and inspectable rather than an LLM-based rewrite, so
  it works with zero API key.

- **Reranking** is a lightweight, multiplicative lexical-overlap/phrase-match
  boost on top of the hybrid score — not a cross-encoder (which would need
  a downloaded model). It nudges chunks that literally contain the query's
  key phrase above chunks that only scored well on diffuse term overlap.

- **Confidence gating for "I don't know."** A candidate answer is only
  generated if the top chunk clears *both* an absolute hybrid-score
  threshold *and* a minimum lexical-overlap-ratio with the query. Two
  independent signals were needed because on a small corpus, a single rare
  shared word (e.g. "year") can make an off-topic query's top match look
  deceptively strong on a single metric alone — see the stock-price example
  in Section 4, which is correctly refused.

- **Conversation / follow-ups** are handled by rewriting elliptical
  follow-ups ("What about international trips?", "And the international
  one?") into a standalone query before retrieval, using either an LLM call
  (if configured) or a heuristic that only fires on actual continuation
  cues (leading "what about", "and…", or a short question containing a
  pronoun like "it"/"that") — not on every short question, so an unrelated
  short question ("Can I expense my gym membership?") isn't wrongly glued
  to the prior turn.

- **Pluggable LLM backend with automatic fallback.** If `ANTHROPIC_API_KEY`
  is set and the `anthropic` package is installed, answers are generated by
  Claude with instructions to cite passage numbers and refuse ungrounded
  claims. If not (or if the API call fails for any reason), the pipeline
  falls back to an extractive mode that surfaces the retrieved passages
  directly. This means the system is gradeable end-to-end with zero setup,
  and upgrades to fluent prose the moment a key is added — no separate code
  path to maintain.

- **Incremental indexing** (`RagPipeline.add_document`) reads and chunks
  only the new file, appends it to the existing in-memory chunk store, and
  rebuilds the lightweight TF-IDF/BM25 statistics — existing documents are
  never re-read from disk. See Trade-offs below for the honest limits of
  this approach.

---

## 6. Assumptions

- The four source policy documents' actual content was not provided, so
  representative sample documents were authored (see Section 1).
- Currency/amounts use INR as a stand-in; the pipeline has no currency
  logic and works with any figures the real documents contain.
- "Sufficient information could not be found" is intentionally shown for
  genuinely out-of-scope questions (stock price, unrelated trivia) — this
  is treated as correct behavior, not a bug, per the assignment's explicit
  requirement.
- A single employee-facing chat session is assumed (no multi-tenant user
  auth); conversation state lives in-process for the session.

## 7. Trade-offs

- **TF-IDF + BM25 vs. dense embeddings:** faster to set up, zero external
  downloads, fully explainable — but weaker than embeddings at genuine
  paraphrase ("time off" vs. "leave") beyond the hand-authored synonym
  table. Swapping in a real embedding model (e.g. `sentence-transformers`
  or an API embedding endpoint) is a scoped change to `indexer.py` /
  `retriever.py` and would directly replace/augment the TF-IDF signal.
- **Incremental indexing is "cheap full rebuild," not true online update:**
  for a corpus of this size (low thousands of chunks), rebuilding TF-IDF/BM25
  statistics over the cached chunk list is milliseconds, so it behaves like
  incremental indexing in practice. It would not scale as-is to a
  multi-million-chunk corpus, where a real vector DB (Pinecone/Weaviate/
  pgvector) with true upsert support would be needed.
- **Confidence gating is heuristic, not learned:** the score + overlap-ratio
  thresholds were tuned against the sample documents and sample questions
  in this repo. On a much larger or more diverse real corpus they would
  need re-tuning (or replacing with a small learned classifier / calibrated
  cross-encoder score).
- **The extractive fallback is not "generation":** without an LLM key, the
  system correctly retrieves and cites the right passages but shows them
  verbatim rather than synthesizing prose. This was a deliberate choice to
  keep the whole pipeline demonstrable and gradeable with zero API cost —
  documented rather than hidden.
- **OCR for scanned PDFs was scoped out:** `document_loader.load_pdf`
  extracts text from text-based PDFs but raises `NotImplementedError` for
  `use_ocr=True`, since a real OCR engine (tesseract, or a hosted OCR API)
  needs a binary/model not guaranteed to exist in every deployment target.
  See Improvements below.

## 8. Improvements I would make with more time

- Wire in a real embedding model (local `sentence-transformers` or a hosted
  embeddings API) as a third retrieval signal alongside TF-IDF/BM25, and
  swap the current lexical rerank for a proper cross-encoder reranker.
- Implement true OCR ingestion (`pytesseract` + `pdf2image`) for scanned
  PDFs, with a confidence score per page so low-quality OCR output can be
  flagged for human review rather than silently indexed.
- Add streaming responses (token-by-token) for the LLM-backed path — the
  Anthropic SDK supports this via `client.messages.stream(...)`; the CLI
  would need to print incrementally instead of waiting for the full
  response.
- Add a lightweight feedback mechanism (👍/👎 per answer, logged with the
  question + retrieved chunk IDs) to build a dataset for tuning the
  confidence thresholds and query-expansion table over time.
- Move the chunk store from a pickle file to a real vector database for
  production-scale documents and true constant-time incremental upserts.
- Add role-based metadata filtering (e.g. contractors only see the subset
  of policy sections that apply to them).
