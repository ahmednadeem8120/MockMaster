# MockMaster — AI-Driven Multimodal Mock Interview Platform

MockMaster is a candidate-facing mock interview system that combines speech recognition, semantic scoring, LLM evaluation, and live behavioural tracking into a single pipeline. Every question is generated dynamically from the candidate's own CV and target job description — no static question banks.

---

## What It Does

Most AI interview tools are built for recruiters. MockMaster is built for candidates. You upload your CV and a job description, answer five spoken questions, and receive a structured report covering semantic accuracy, technical depth, and non-verbal behaviour — all scored transparently.

The composite scoring model combines 60% LLM qualitative judgment with 40% SBERT semantic similarity. Across a 60-response evaluation dataset (20 test cases × 3 answer tiers), the system achieved a **5.78/10 discrimination gap** between strong and irrelevant answers, a Pearson correlation of **r = 0.980** against a human domain expert evaluator, and a Mann-Whitney U significance of **p = 3.37×10⁻⁸**.

---

## Core Stack

| Component | Purpose |
|---|---|
| Llama 3 (via Ollama) | LLM scoring and feedback generation |
| all-MiniLM-L6-v2 (SBERT) | Semantic similarity scoring |
| BAAI/bge-small-en-v1.5 | RAG retrieval embeddings |
| FAISS + LangChain | Vector store and RAG pipeline |
| GLiNER | CV entity extraction (NER) |
| OpenAI Whisper | Speech-to-text transcription |
| MediaPipe | Live behavioural tracking (eye contact, posture, gestures, blink rate) |
| FastAPI | Backend API |
| React | Frontend interface |

Hardware tested on: Apple MacBook Air M4 (built-in webcam and microphone).

---

## Project Structure

```
Mock Master/
├── api.py                          # FastAPI backend — scoring, RAG, report generation
├── ingest.py                       # CV ingestion and FAISS index builder
├── ner_extractor.py                # GLiNER-based CV entity extraction
├── behavioral_analyzer.py          # MediaPipe behavioural tracking
├── evaluator_formula_test_quantitative.py  # Formula benchmarking script
├── statistical_significance_tests.py       # Pearson, Spearman, Mann-Whitney U
├── human_evaluation_results.py     # Visualisation generation
├── App.jsx                         # React frontend
├── requirements.txt                # Python dependencies
├── data/
│   └── formula_test_results.json   # Benchmark output (60 responses, 11 formulas)
└── output_visuals/                 # Generated evaluation charts
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Ollama with Llama 3

```bash
ollama run llama3
```

### 3. Ingest your CV

Place your CV at `data/cv.pdf` and your job description at `data/job_description.txt`, then run:

```bash
python ingest.py
```

### 4. Start the backend

```bash
uvicorn api:app --reload
```

### 5. Start the frontend

```bash
cd frontend
npm install
npm start
```

---

## Scoring Formula

```
Composite Score = (0.6 × LLM Score) + (0.4 × SBERT Score)
```

Both components are scored on a 0–10 scale. The behavioural modifier (MediaPipe) applies a live adjustment to the final composite based on eye contact, head posture, hand gesture activity, blink rate, and enthusiasm level. In evaluation, this modifier produced a mean adjustment of **+1.65 points**.

---

## Evaluation Results

| Metric | Value |
|---|---|
| Pearson r (F5 vs Human) | 0.980 |
| Discrimination gap (Strong vs Irrelevant) | 4.18 / 10 |
| Mann-Whitney U p-value | 3.37×10⁻⁸ |
| Evaluation dataset | 60 responses (20 cases × 3 tiers) |
| Behavioural modifier mean effect | +1.90 pts |

---

## Acknowledgements

Final Year Project — BSc Computer Science, University of West London (RAK Campus), AY 2025–26.
Supervised by Dr H. Shaheen

---

## Author

Ahmed Nadeem — 32146990@student.uwl.ac.uk
