import os
import shutil
import whisper
import json
import re
import asyncio
from functools import partial
from fastapi.responses import StreamingResponse
from behavioral_analyzer import generate_video_frames, reset_session, stop_session_and_wait
from ingest import build_knowledge_base
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from sentence_transformers import SentenceTransformer, util
 
app = FastAPI(title="MockMaster Backend")
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# --- Global State ---
# NOTE ON CONCURRENCY:
# This backend uses module-level globals (current_question, transcript,
# jd_text, cv_summary, behavioral_metrics.json) to hold session state.
# This is a deliberate design choice for a single-user FYP demo — it keeps
# the code simple and keeps the focus on the multimodal scoring pipeline.
# Consequence: only ONE interview can run at a time. Two concurrent users
# would have their questions, answers, and scores overwrite each other.
# A production deployment would need per-session state keyed by a session_id
# (e.g. Redis, an in-memory dict keyed by UUID, or FastAPI dependencies).
# This is listed as "future work" in the project report.
current_question = ""
current_ideal_answer = ""          # Stores the ideal response for SBERT comparison
transcript = []
vector_db = None
llm = None
whisper_model = None
sbert_model = None
target_questions = 5
current_difficulty = "Medium"
jd_text = ""          # Raw job description text — used for JD-focused question gen + fit scoring
cv_summary = ""       # Short CV NER profile text — used for fit scoring in /end report
 
 
def extract_json(text):
    """Bulletproof JSON extractor to handle LLMs that add markdown or filler text."""
    try:
        match = re.search(r'\{.*\}', text.replace('\n', ''), re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(text)
    except Exception:
        return {
            "feedback": "Processed.",
            "score": 5,
            "next_question": "Let's move on. Could you expand on your technical experience?"
        }
 
 
def compute_sbert_similarity(candidate_answer: str, ideal_answer: str) -> float:
    """
    Computes cosine similarity between the candidate's answer and the ideal
    response outline using Sentence-BERT. Returns a float in [0.0, 1.0].
    """
    if not candidate_answer.strip() or not ideal_answer.strip():
        return 0.0
    embeddings = sbert_model.encode(
        [candidate_answer, ideal_answer],
        convert_to_tensor=True
    )
    similarity = util.cos_sim(embeddings[0], embeddings[1])
    return round(float(max(0.0, similarity.item())), 3)
 
 
@app.on_event("startup")
def load_system():
    global vector_db, llm, whisper_model, sbert_model
    print("Booting MockMaster Engine...")
 
    whisper_model = whisper.load_model("base.en")
    print("Whisper loaded.")
 
    # 'all-MiniLM-L6-v2' is the standard model for semantic textual similarity —
    # fast inference, strong performance, exactly what the proposal specified.
    print("Loading SBERT model (all-MiniLM-L6-v2)...")
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("SBERT loaded.")
 
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vector_db = FAISS.load_local("./faiss_db", embeddings, allow_dangerous_deserialization=True)
    print("FAISS vector DB loaded.")
 
    llm = OllamaLLM(model="llama3", temperature=0.2, format="json")
    print("System fully online.")
 
 
@app.post("/upload")
async def upload_documents(cv: UploadFile = File(...), jd: UploadFile = File(...)):
    global vector_db, jd_text, cv_summary
    os.makedirs("data", exist_ok=True)

    with open("data/cv.pdf", "wb") as buffer:
        shutil.copyfileobj(cv.file, buffer)
    with open("data/job_description.txt", "wb") as buffer:
        shutil.copyfileobj(jd.file, buffer)

    print("Rebuilding Knowledge Base with new user data...")
    build_knowledge_base()

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vector_db = FAISS.load_local("./faiss_db", embeddings, allow_dangerous_deserialization=True)

    # Cache raw JD and CV NER profile for /start prompts and /end fit scoring
    try:
        with open("data/job_description.txt", "r", encoding="utf-8", errors="ignore") as f:
            jd_text = f.read().strip()
    except Exception:
        jd_text = ""

    try:
        with open("data/ner_extracted_profile.txt", "r", encoding="utf-8", errors="ignore") as f:
            cv_summary = f.read().strip()
    except Exception:
        cv_summary = ""

    print("Dynamic Knowledge Base online.")
    return {"status": "success", "message": "Knowledge base synchronized!"}
 
 
@app.get("/start")
def start_interview(difficulty: str = "Medium", questions: int = 5):
    global current_question, current_ideal_answer, transcript, target_questions, current_difficulty
    transcript = []
    target_questions = int(questions)
    current_difficulty = difficulty

    # Clear any stale stop signal and mark the start time for the behavioural
    # tracker. Without this, a second interview in the same process run would
    # inherit the previous session's stop_event and exit immediately.
    reset_session()
 
    # JD chunks dominate (k=3). CV chunk (k=1) used only to personalise wording.
    jd_results = vector_db.similarity_search(
        "job responsibilities requirements qualifications duties skills needed for this role", k=3
    )
    cv_results = vector_db.similarity_search("candidate background education experience", k=1)
    jd_context = "\n".join([doc.page_content for doc in jd_results])
    cv_context = "\n".join([doc.page_content for doc in cv_results])

    prompt = f"""
    You are an interviewer assessing whether this candidate can perform the role in the job description.
    Your goal is to test fit for THIS ROLE — not to quiz the candidate on their existing background.

    JOB DESCRIPTION (PRIMARY — base your question entirely on what this role requires):
    {jd_context}

    CANDIDATE BACKGROUND (SECONDARY — use only to personalise wording, never make it the subject):
    {cv_context}

    Generate ONE {current_difficulty}-level interview question that directly tests whether the candidate
    can meet a requirement of THIS JOB DESCRIPTION.

    CRITICAL RULES:
    - The question MUST target a skill, responsibility, or scenario from the JOB DESCRIPTION.
    - If the candidate's background is in a completely different field, deliberately ask about what
      the JOB needs. For example: JD = delivery driver, CV = software engineer → ask about route
      planning, time pressure, customer interaction, or physical logistics. NOT about coding.
    - The question MUST be answerable verbally. No code, no diagrams, no keyboard tasks.
    - Good frames: "How would you handle...", "Describe a time you...", "What would you do if...",
      "What experience do you have with..."

    Respond ONLY with a JSON object in this exact format:
    {{
        "question": "The interview question text here",
        "ideal_answer": "A concise 3-5 sentence model answer a strong candidate for THIS ROLE would give verbally"
    }}
    """
    raw_response = llm.invoke(prompt)
    parsed_data = extract_json(raw_response)
    current_question = parsed_data.get("question", "To begin, can you tell me about the most challenging project you have worked on?")
    current_ideal_answer = parsed_data.get("ideal_answer", "")
 
    print(f"Question ready. Ideal answer generated: {'Yes' if current_ideal_answer else 'No'}")
    return {"question": current_question}
 
 
@app.post("/reply")
async def process_reply(audio: UploadFile = File(...)):
    global current_question, current_ideal_answer, transcript, vector_db, target_questions, current_difficulty
 
    temp_file = f"temp_{audio.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)
 
    print("Transcribing audio...")
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        partial(
            whisper_model.transcribe,
            temp_file,
            word_timestamps=True,
            initial_prompt="Umm, let me think... uhh, okay. Like, yeah."
        )
    )
    user_text = result["text"].strip()
    os.remove(temp_file)

    # --- 1. Filler word count ---
# Regex catches all elongated Whisper variants (um/umm/ummm, uh/uhh/uhhh etc.)
    FILLER_PATTERNS = re.compile(
    r'\b(um+|uh+|er+|ah+|hm+|mhm+|like)\b',
    re.IGNORECASE
)
    filler_count = len(FILLER_PATTERNS.findall(user_text))
 
    # --- 2. Long pause count ---
    long_pauses = 0
    for segment in result.get("segments", []):
        words = segment.get("words", [])
        for i in range(len(words) - 1):
            if words[i + 1]["start"] - words[i]["end"] > 1.2:
                long_pauses += 1
 
    print(f"Fluency: {filler_count} fillers, {long_pauses} pauses.")
 
    # --- 3. Answer quality gate ---
    # Detect garbage/off-topic answers before SBERT scoring.
    # A legitimate interview answer needs at least 15 words.
    word_count = len(user_text.split())
    is_too_short = word_count < 15
    print(f"Answer word count: {word_count} words.")
 
    # --- 4. SBERT Semantic Similarity Score ---
    semantic_score = 0.0
    if current_ideal_answer:
        semantic_score = compute_sbert_similarity(user_text, current_ideal_answer)
        print(f"SBERT Semantic Similarity: {semantic_score:.3f}")
    else:
        print("No ideal answer available — skipping SBERT scoring.")
 
    # Hard floor: if answer is too short OR SBERT is extremely low,
    # cap the semantic score to prevent inflated composite scores.
    # A 0.67 SBERT on a 2-sentence garbage answer is an embedding
    # coincidence, not genuine understanding.
    if is_too_short and semantic_score > 0.4:
        semantic_score = min(semantic_score, 0.35)
        print(f"Answer too short — SBERT capped to {semantic_score:.3f}")
 
    # --- 5. LLM Qualitative Evaluation ---
    docs = vector_db.similarity_search(user_text, k=2)
    dynamic_context = "\n".join([doc.page_content for doc in docs])
 
    prompt = f"""
You are a strict technical interviewer conducting a spoken voice interview.
You must evaluate the candidate's answer based purely on your own judgement.
You have NOT been given any semantic similarity score — judge the content yourself.

Question asked: "{current_question}"
Ideal answer outline: "{current_ideal_answer}"
Candidate answered: "{user_text}"
Answer word count: {word_count} words.
Speech Analytics: {filler_count} filler words, {long_pauses} hesitations over 1.2 seconds.
Relevant context from candidate CV and job description: {dynamic_context}

SCORING RULES — follow these strictly:
- Score ONLY on technical accuracy, specificity, and depth of the answer itself.
- 1-2: Fewer than 15 words, or completely off-topic with no relevant content.
- 3-4: Correct topic but vague — no specific methods, tools, steps, or examples named.
- 5-6: Covers the key idea but lacks depth or misses important technical details.
- 7-8: Strong answer — specific, accurate, mentions concrete approaches or tools.
- 9-10: Exceptional — technically deep, specific examples, clearly demonstrates expertise.
- Be strict. Most interview answers should score 4-6. Reserve 7+ for genuinely strong responses.
- Do NOT reward an answer just because it uses the right words without explaining them.

NEXT QUESTION RULES — this is critical:
- The next question MUST test a requirement from the JOB DESCRIPTION, not the candidate's prior background.
- If the candidate's CV background differs from the role, keep probing what the JOB requires.
- The next question MUST be answerable verbally. No code, no diagrams, no keyboard tasks.
- Good frames: "How would you handle...", "Describe a time...", "What would you do if..."
- JOB DESCRIPTION CONTEXT for next question: {jd_text[:600]}

Output a JSON object with EXACTLY these three keys:
"feedback": A 2-3 sentence evaluation of the answer. Reference the word count and identify specific gaps. Do not mention semantic similarity scores.
"llm_score": An integer 1-10 following the scoring rules above strictly.
"next_question": A nested object: {{"question": "Next verbal question text", "ideal_answer": "3-5 sentence model answer"}}
"""
 
    raw_response = await loop.run_in_executor(None, llm.invoke, prompt)
    parsed_data = extract_json(raw_response)
 
    feedback = parsed_data.get("feedback", "Thank you for the response.")
    try:
        llm_score = int(parsed_data.get("llm_score", 5))
    except (ValueError, TypeError):
        llm_score = 5
 
    # Hard cap LLM score for very short answers — LLM sometimes ignores
    # the instruction and is still too generous
    if is_too_short:
        llm_score = min(llm_score, 2)
        print(f"Answer too short — LLM score capped to {llm_score}")
 
    # --- 6. Composite Final Score ---
    # 60% LLM qualitative judgment + 40% SBERT semantic similarity.
    # LLM scoring rules produce [1,10], not [0,10]. Normalising to [0,10] so
    # that a complete non-answer (llm=1, sbert=0) correctly scores 0, not 0.6.
    sbert_as_10 = round(semantic_score * 10, 1)
    llm_as_10   = round((llm_score - 1) / 9 * 10, 1)
    final_score = round((llm_as_10 * 0.6) + (sbert_as_10 * 0.4), 1)
    print(f"Scores — LLM: {llm_score}/10 (norm {llm_as_10}) | SBERT: {sbert_as_10}/10 | Final: {final_score}/10 | Words: {word_count}")
 
    # --- 7. Extract next question + its ideal answer ---
    is_complete = False
    if len(transcript) + 1 >= target_questions:
        next_q = "Thank you. We have completed the allocated questions for this session. Please terminate the feed to generate your report."
        next_ideal = ""
        is_complete = True
    else:
        raw_next = parsed_data.get("next_question", {})
        if isinstance(raw_next, dict):
            next_q = raw_next.get("question", f"Could you elaborate on: {dynamic_context[:80]}...?")
            next_ideal = raw_next.get("ideal_answer", "")
        else:
            next_q = str(raw_next) if raw_next else f"Could you elaborate on: {dynamic_context[:80]}...?"
            next_ideal = ""
 
    answered_question = current_question
    current_question = next_q
    current_ideal_answer = next_ideal
 
    transcript.append({
        "question": answered_question,
        "answer": user_text,
        "feedback": feedback,
        "score": final_score,
        "llm_score": llm_score,
        "semantic_score": semantic_score,
        "filler_count": filler_count,
        "long_pauses": long_pauses,
        "word_count": word_count
    })
 
    return {
        "transcription": user_text,
        "feedback": feedback,
        "score": final_score,
        "llm_score": llm_score,
        "semantic_score": semantic_score,
        "next_question": next_q,
        "is_complete": is_complete
    }
 
 
 
 
@app.get("/end")
def generate_report():
    global transcript
 
    if not transcript:
        return {"error": "No data"}

    # Tell the video generator to exit its capture loop and flush final metrics
    # to disk. Without this, the read below would pick up the mid-session
    # `recording` snapshot written every 30 frames, not the true finals.
    print("Stopping behavioral tracker and waiting for finalised metrics...")
    completed_cleanly = stop_session_and_wait(timeout=3.0)
    if not completed_cleanly:
        print("  [WARN] Behavioral metrics did not reach 'completed' status within timeout.")
 
    # ------------------------------------------------------------------ #
    #  1. Collect all data                                                 #
    # ------------------------------------------------------------------ #
    behavioral_data = {
        "eye_contact_score": 0,
        "head_posture_score": 0,
        "enthusiasm_score": 0,
        "hand_gesture_score": 0,
        "blink_count": 0
    }
    if os.path.exists("behavioral_metrics.json"):
        try:
            with open("behavioral_metrics.json", "r") as f:
                behavioral_data = json.load(f)
        except Exception as e:
            print(f"Error reading behavioral metrics: {e}")
 
    avg_technical = round(sum(item["score"] for item in transcript) / len(transcript), 1)
    avg_semantic  = round(sum(item.get("semantic_score", 0) for item in transcript) / len(transcript), 3)
    avg_llm       = round(sum(item.get("llm_score", 0) for item in transcript) / len(transcript), 1)
    total_fillers = sum(item.get("filler_count", 0) for item in transcript)
    total_pauses  = sum(item.get("long_pauses", 0) for item in transcript)
    avg_words     = round(sum(item.get("word_count", 0) for item in transcript) / len(transcript), 1)
 
    print(f"Session Averages — Technical: {avg_technical} | Semantic: {avg_semantic:.3f} | LLM: {avg_llm}")
 
    # ------------------------------------------------------------------ #
    #  2. Behavioral signal fusion                                         #
    #                                                                      #
    #  Each signal scored against a realistic ideal range.                #
    #  Perfect scores are penalised — they indicate unnatural behaviour.  #
    #  Behavioral modifier = bonus signals + stress penalty.              #
    #  Applied to the technical average to produce the adjusted final.    #
    # ------------------------------------------------------------------ #
 
    eye  = behavioral_data.get("eye_contact_score", 0)
    post = behavioral_data.get("head_posture_score", 0)
    enth = behavioral_data.get("enthusiasm_score", 0)
    gest = behavioral_data.get("hand_gesture_score", 0)
    blink_count  = behavioral_data.get("blink_count", 0)
    blinks_per_q = blink_count / max(1, len(transcript))
 
    # --- Eye contact: ideal 55–90%, natural conversational range ---
    # Below 55% = avoidant. Above 90% = unnatural fixed stare.
    if 55 <= eye <= 90:
        eye_bonus = 1.0
        eye_label = f"{round(eye)}% — Natural range"
    elif 40 <= eye < 55 or 90 < eye <= 95:
        eye_bonus = 0.3
        eye_label = f"{round(eye)}% — Slightly outside ideal"
    else:
        eye_bonus = -0.5
        eye_label = f"{round(eye)}% — Avoidant or fixed stare"
 
    # --- Head posture: ideal 50–95%, robotically rigid above 95% ---
    if 50 <= post <= 95:
        post_bonus = 0.75
        post_label = f"{round(post)}% — Good posture maintained"
    elif 35 <= post < 50 or 95 < post <= 100:
        post_bonus = 0.25
        post_label = f"{round(post)}% — Slightly outside ideal"
    else:
        post_bonus = -0.25
        post_label = f"{round(post)}% — Poor posture throughout"
 
    # --- Enthusiasm: ideal 5–35%, professional warmth without over-performance ---
    # Below 8% = cold/anxious. Above 35% = nervous over-smiling.
    if 8 <= enth <= 35:           # true ideal range
     enth_bonus = 0.5
     enth_label = f"{round(enth)}% — Appropriate warmth"
    elif 5 <= enth < 8:           # slightly cold — now actually reachable
     enth_bonus = 0.1
     enth_label = f"{round(enth)}% — Slightly cold"
    elif 35 < enth <= 50:         # slight over-smiling
     enth_bonus = 0.1
     enth_label = f"{round(enth)}% — Slightly outside ideal"
    else:                          # < 5% or > 50%
     enth_bonus = -0.25
     enth_label = f"{round(enth)}% — Too cold or excessive smiling"
 
    # --- Hand gestures: ideal 5–40%, moderate use aids explanation ---
    # Zero gestures = rigid. Above 40% = distracting or nervous.
    if 5 <= gest <= 40:
        gest_bonus = 0.5
        gest_label = f"{round(gest)}% — Effective use of gestures"
    elif 0 <= gest < 5 or 40 < gest <= 60:
        gest_bonus = 0.1
        gest_label = f"{round(gest)}% — Slightly outside ideal"
    else:
        gest_bonus = -0.1
        gest_label = f"{round(gest)}% — Excessive or absent gestures"
 
    # --- Blink rate: normalised by session duration where possible --------
    # Human baseline is 12-20 blinks/minute. Using blinks-per-question is
    # misleading because a 2-minute answer mechanically produces more blinks
    # than a 30-second answer. If session_duration_sec is available from the
    # behavioural tracker we normalise per minute; otherwise we fall back to
    # per-question. Penalty-only: calm blinking is neutral, not rewarded.
    session_duration_sec = behavioral_data.get('session_duration_sec', 0)
    if session_duration_sec and session_duration_sec >= 10:
        blinks_per_min = round((blink_count / session_duration_sec) * 60, 1)
        blink_rate_display = f"{blinks_per_min}/min"
        blink_rate_ideal = "12–20/min baseline"
        if blinks_per_min <= 20:
            blink_penalty = 0.0
            stress_label = "Composed and Calm"
            stress_color = "#059669"
        elif blinks_per_min <= 35:
            blink_penalty = -0.25
            stress_label = "Moderate Stress"
            stress_color = "#d97706"
        else:
            blink_penalty = -0.75
            stress_label = "High Stress Detected"
            stress_color = "#dc2626"
    else:
        # Fallback for sessions where duration was not captured.
        blink_rate_display = f"{round(blinks_per_q, 1)}/q"
        blink_rate_ideal = "<8/q ideal"
        if blinks_per_q <= 8:
            blink_penalty = 0.0
            stress_label = "Composed and Calm"
            stress_color = "#059669"
        elif blinks_per_q <= 15:
            blink_penalty = -0.25
            stress_label = "Moderate Stress"
            stress_color = "#d97706"
        else:
            blink_penalty = -0.75
            stress_label = "High Stress Detected"
            stress_color = "#dc2626"
 
    # --- Compute total behavioral modifier ---
    behavioral_bonus    = eye_bonus + post_bonus + enth_bonus + gest_bonus
    behavioral_modifier = round(behavioral_bonus + blink_penalty, 2)
 
    # --- Apply to technical average and clamp to 0–10 ---
    avg_final = round(max(0.0, min(10.0, avg_technical + behavioral_modifier)), 1)
 
    print(f"Behavioral Fusion — Eye: {eye_bonus:+.2f} | Posture: {post_bonus:+.2f} | "
          f"Enthusiasm: {enth_bonus:+.2f} | Gestures: {gest_bonus:+.2f} | "
          f"Blink penalty: {blink_penalty:+.2f} | Total modifier: {behavioral_modifier:+.2f}")
    print(f"Technical avg: {avg_technical} → Adjusted final: {avg_final}")
 
    # ------------------------------------------------------------------ #
    #  3. Verdict label (based on adjusted final)                         #
    # ------------------------------------------------------------------ #
    if avg_final >= 8.0:
        verdict_label = "Exceptional Candidate"
        verdict_color = "#059669"
    elif avg_final >= 6.5:
        verdict_label = "Strong Candidate — Above Average"
        verdict_color = "#0284c7"
    elif avg_final >= 5.0:
        verdict_label = "Competent Candidate — Meets Baseline"
        verdict_color = "#d97706"
    else:
        verdict_label = "Needs Development — Below Expectations"
        verdict_color = "#dc2626"
 
    # ------------------------------------------------------------------ #
    #  4. Transcript text for LLM                                         #
    # ------------------------------------------------------------------ #
    transcript_text = ""
    for i, item in enumerate(transcript):
        transcript_text += f"Q{i+1}: {item['question']}\n"
        transcript_text += f"Answer ({item.get('word_count', '?')} words): {item['answer']}\n"
        transcript_text += f"SBERT: {item.get('semantic_score', 0):.2f} | LLM: {item.get('llm_score', 0)}/10 | Composite: {item['score']}/10\n"
        transcript_text += f"Fillers: {item.get('filler_count', 0)} | Hesitations: {item.get('long_pauses', 0)}\n\n"
 
    # ------------------------------------------------------------------ #
    #  5. LLM writes ONLY the four deep prose sections                    #
    #                                                                      #
    #  Pre-compute calibrated assessments for each signal so the LLM      #
    #  cannot describe a signal as negative when the quantitative table   #
    #  shows it is in the ideal range. This prevents the report from     #
    #  contradicting itself and stops the LLM inventing gaps that do     #
    #  not exist in the data.                                             #
    # ------------------------------------------------------------------ #

    # Filler word calibration — aligned with typical interviewer tolerance.
    avg_fillers_per_q = round(total_fillers / max(1, len(transcript)), 1)
    if total_fillers <= 2:
        filler_assessment = f"{total_fillers} filler words total is minimal and within normal professional range (≤2 is not a concern)."
    elif total_fillers <= 5:
        filler_assessment = f"{total_fillers} filler words total is slightly elevated but still acceptable for a spoken interview."
    elif avg_fillers_per_q <= 3:
        filler_assessment = f"{total_fillers} filler words ({avg_fillers_per_q} per question) is noticeable and worth reducing through practice."
    else:
        filler_assessment = f"{total_fillers} filler words ({avg_fillers_per_q} per question) is excessive and materially undermines professional delivery."

    # Hesitation calibration.
    if total_pauses == 0:
        pause_assessment = "No long hesitations detected — the candidate spoke with composed pacing."
    elif total_pauses <= 2:
        pause_assessment = f"{total_pauses} long hesitations is within normal thinking-pause range."
    else:
        pause_assessment = f"{total_pauses} long hesitations suggests the candidate was searching for words under pressure."

    # SBERT vs LLM agreement — only flag a gap when the scores actually diverge.
    sbert_on_10 = avg_semantic * 10
    score_diff = abs(sbert_on_10 - avg_llm)
    if avg_semantic < 0.35 and avg_llm <= 3:
        agreement_note = ("Both SBERT and LLM scores are very low and in close agreement — "
                          "the system is confident the answers were off-topic or missing, NOT that there is a 'gap' "
                          "between surface communication and deep knowledge. Do NOT describe a gap here.")
    elif avg_semantic >= 0.55 and avg_llm >= 6 and score_diff < 2.0:
        agreement_note = ("Both SBERT and LLM scores are reasonable and in close agreement — "
                          "the answers were substantive. Do NOT describe a gap between knowledge and communication.")
    elif score_diff >= 2.5:
        if avg_llm > sbert_on_10:
            agreement_note = ("LLM score is materially HIGHER than SBERT — the candidate communicated confidently "
                              "but their content diverged from the ideal answer content.")
        else:
            agreement_note = ("SBERT score is materially HIGHER than LLM — the candidate's wording overlapped with "
                              "the ideal answer but the LLM judged the reasoning or depth as weak.")
    else:
        agreement_note = "SBERT and LLM scores are broadly consistent with each other."

    # Behavioral signal assessments — use the EXACT same ideal ranges as the
    # quantitative table so the prose and the table cannot contradict each other.
    eye_val = behavioral_data.get('eye_contact_score', 0)
    post_val = behavioral_data.get('head_posture_score', 0)
    enth_val = behavioral_data.get('enthusiasm_score', 0)
    gest_val = behavioral_data.get('hand_gesture_score', 0)

    eye_assess  = f"{eye_val}% eye contact — {'inside ideal 55-90% range, describe POSITIVELY as engaged and natural' if 55 <= eye_val <= 90 else 'outside ideal range, describe as needing adjustment'}."
    post_assess = f"{post_val}% head posture — {'inside ideal 50-95% range, describe POSITIVELY as attentive' if 50 <= post_val <= 95 else 'outside ideal range, describe as needing adjustment'}."
    enth_assess = f"{enth_val}% enthusiasm — {'inside ideal 8-35% range, describe POSITIVELY as appropriate professional warmth' if 8 <= enth_val <= 35 else ('too low, describe as cold or anxious' if enth_val < 8 else 'too high, describe as nervous over-smiling')}."
    gest_assess = f"{gest_val}% hand gestures — {'inside ideal 5-40% range, describe POSITIVELY as effective' if 5 <= gest_val <= 40 else 'outside ideal range, describe as needing adjustment'}."

    # Blink rate — calibrate against session duration if available, else per-question.
    session_duration_sec = behavioral_data.get('session_duration_sec', 0)
    if session_duration_sec and session_duration_sec > 10:
        blinks_per_min = round((blink_count / session_duration_sec) * 60, 1)
        # Normal human blink rate is 12-20/min. Anything inside that is baseline, NOT stress.
        if blinks_per_min <= 20:
            blink_assess = f"{blinks_per_min} blinks/minute — within normal human baseline (12-20/min). Do NOT describe as stress."
        elif blinks_per_min <= 35:
            blink_assess = f"{blinks_per_min} blinks/minute — elevated above baseline, suggests moderate cognitive load."
        else:
            blink_assess = f"{blinks_per_min} blinks/minute — significantly elevated, suggests high stress."
    else:
        # Fall back to per-question rate using the existing stress label.
        blink_assess = f"{round(blinks_per_q,1)} blinks per question — stress level: {stress_label}."

    print("AI generating deep prose analysis...")
    prose_prompt = f"""
You are a senior technical recruiter writing a professional interview evaluation report.
Write FOUR analytical paragraphs — one per section — of 6-8 sentences each.
Reference the specific numbers and questions. Be specific and actionable.

CRITICAL CALIBRATION RULES — you MUST follow these or the report will contradict its own data:

1. The quantitative tables in this report show which signals are in their "ideal range".
   If a signal is described below as "inside ideal range, describe POSITIVELY", you MUST NOT
   describe it negatively, regardless of the overall composite score. The composite score
   does not override individual signal assessments.

2. Do NOT invent a "gap" between SBERT and LLM scores unless one exists. Read the
   agreement note below carefully and follow it exactly.

3. Do NOT escalate language beyond what the numbers support. Two filler words is not
   "unacceptable". A blink rate inside normal human baseline is not "stress".

4. Interpret values ONLY using the assessments pre-computed below. Do not impose your
   own thresholds.

PRE-COMPUTED SIGNAL ASSESSMENTS (use these verbatim as your interpretation):
- Filler words: {filler_assessment}
- Hesitations: {pause_assessment}
- SBERT vs LLM agreement: {agreement_note}
- Eye contact: {eye_assess}
- Head posture: {post_assess}
- Enthusiasm: {enth_assess}
- Hand gestures: {gest_assess}
- Blink rate: {blink_assess}

QUANTITATIVE DATA:
- Composite Score: {avg_final}/10 — {verdict_label}
- SBERT Semantic Similarity: {avg_semantic:.2f}/1.0
  (Scale: 0.8+ = excellent, 0.55-0.8 = acceptable, 0.35-0.55 = vague, below 0.35 = off-topic)
- LLM Qualitative Score: {avg_llm}/10
- Average Words Per Answer: {avg_words} (professional interview answers should be 60-120 words; below 15 = non-answer)
- Session length: {session_duration_sec}s

FULL TRANSCRIPT WITH ALL SCORES:
{transcript_text}

JOB DESCRIPTION (the role being applied for):
{jd_text[:800]}

CANDIDATE PROFILE FROM CV:
{cv_summary[:600]}

SPECIAL CASE — HOSTILE OR REFUSED ANSWERS:
If any answer contains profanity, refusal to engage, or clearly non-responsive content
(e.g. "fuck off", "I don't know", a single shrug word, nonsense), the correct interpretation
is that the candidate did NOT engage with the question, not that they lack knowledge.
The action plan in this case must focus on engagement and preparation, not technical study.

Respond ONLY with a JSON object with exactly these FIVE keys.
Each value is plain text with NO html tags:
{{
  "technical_analysis": "6-8 sentences. Identify which questions scored lowest and what that reveals. Use the SBERT-vs-LLM agreement note above verbatim — do not invent a gap. If both scores are low and in agreement, say so and explain what the shared weakness indicates. Name specific topics from the lowest-scoring questions that need study.",

  "communication_analysis": "6-8 sentences. Use the filler and hesitation assessments above verbatim — do not escalate. Discuss the average answer length of {avg_words} words against the 60-120 word professional benchmark. If any answer was under 15 words, flag it as a non-answer. Discuss what the fluency profile indicates about preparation.",

  "behavioral_analysis": "6-8 sentences. Use the eye-contact, posture, enthusiasm, gesture, and blink assessments above verbatim. If a signal is flagged as 'inside ideal range, describe POSITIVELY', describe it positively even if other signals are weak. Be honest about which signals need work and which are already strong.",

  "action_plan": "6-8 sentences ranked by impact. If any answer was non-responsive or hostile, the top priority is 'engage with the question' not 'study technical topics' — preparation and mindset must come before content. Give specific, concrete practice steps (not 'study more'). Include a realistic 4-week timeline and a realistic projected composite score from the current baseline of {avg_final}/10.",

  "job_fit": {{
    "score": <integer 0-100 for how well the candidate fits THIS job description>,
    "verdict": "<one of: Excellent Fit | Good Fit | Partial Fit | Low Fit | Mismatched Role>",
    "analysis": "6-8 sentences. Compare JD requirements against the candidate's actual background. If the role is outside their background, say so honestly. Explain which requirements they can meet, which they cannot, and whether transferable skills bridge the gap. End with a direct recommendation: proceed, proceed with caution, or look elsewhere."
  }}
}}
"""
 
    html_writer_llm = OllamaLLM(model="llama3", temperature=0.3, format="json")
    raw_prose = html_writer_llm.invoke(prose_prompt)
 
    try:
        prose = json.loads(raw_prose)
    except Exception:
        prose = {
            "technical_analysis": "Technical analysis unavailable.",
            "communication_analysis": "Communication analysis unavailable.",
            "behavioral_analysis": "Behavioral analysis unavailable.",
            "action_plan": "Please review your individual question scores above for improvement areas.",
            "job_fit": {"score": 50, "verdict": "Assessment Unavailable", "analysis": "Job fit analysis could not be generated."}
        }

    tech_text = prose.get("technical_analysis", "")
    comm_text = prose.get("communication_analysis", "")
    beh_text  = prose.get("behavioral_analysis", "")
    plan_text = prose.get("action_plan", "")

    # --- Job Fit extraction ---
    raw_fit       = prose.get("job_fit", {})
    if not isinstance(raw_fit, dict):
        raw_fit = {}
    fit_score     = int(raw_fit.get("score", 50))
    fit_score     = max(0, min(100, fit_score))   # clamp 0-100
    # Override LLM verdict with score-anchored verdict to ensure consistency
    if fit_score >= 80:
        fit_verdict = "Excellent Fit"
    elif fit_score >= 65:
        fit_verdict = "Good Fit"
    elif fit_score >= 50:
        fit_verdict = "Partial Fit"
    else:
        fit_verdict = "Low Fit"
    fit_analysis  = raw_fit.get("analysis", "Job fit analysis unavailable.")

    # Colour theme for fit score
    if fit_score >= 75:
        fit_color = "#059669"; fit_bg = "#d1fae5"; fit_ring = "#6ee7b7"
    elif fit_score >= 50:
        fit_color = "#d97706"; fit_bg = "#fef3c7"; fit_ring = "#fcd34d"
    else:
        fit_color = "#dc2626"; fit_bg = "#fee2e2"; fit_ring = "#fca5a5"
 
    # ------------------------------------------------------------------ #
    #  6. HTML helpers                                                     #
    # ------------------------------------------------------------------ #
    def bar(label, value, max_val=100, color="#1D9E75", suffix="%"):
        pct = min(100, round((value / max_val) * 100))
        return f"""
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">
          <span style="width:160px;font-size:13px;color:#475569;flex-shrink:0;">{label}</span>
          <div style="flex:1;height:8px;background:#e2e8f0;border-radius:4px;overflow:hidden;">
            <div style="width:{pct}%;height:100%;background:{color};border-radius:4px;"></div>
          </div>
          <span style="font-size:13px;font-weight:600;color:#1e293b;width:60px;text-align:right;">{round(value)}{suffix}</span>
        </div>"""
 
    def badge(text, color, bg):
        return f'<span style="background:{bg};color:{color};font-size:11px;font-weight:600;padding:3px 10px;border-radius:4px;margin-right:6px;">{text}</span>'
 
    def score_color(score, max_score=10):
        ratio = score / max_score
        if ratio >= 0.75: return "#059669", "#d1fae5"
        if ratio >= 0.5:  return "#d97706", "#fef3c7"
        return "#dc2626", "#fee2e2"
 
    def sbert_label(s):
        if s >= 0.75: return "Excellent", "#059669", "#d1fae5"
        if s >= 0.55: return "Acceptable", "#0284c7", "#dbeafe"
        if s >= 0.35: return "Vague", "#d97706", "#fef3c7"
        return "Off-Topic", "#dc2626", "#fee2e2"
 
    # ------------------------------------------------------------------ #
    #  7. Per-question cards                                               #
    # ------------------------------------------------------------------ #
    q_cards_html = ""
    for i, item in enumerate(transcript):
        comp   = item["score"]
        sbert  = item.get("semantic_score", 0)
        sbert_pct = round(sbert * 100)
        llm_s  = item.get("llm_score", 0)
        fills  = item.get("filler_count", 0)
        pauses = item.get("long_pauses", 0)
        words  = item.get("word_count", 0)
        fb     = item.get("feedback", "")
        q_txt  = item.get("question", "")
        a_txt  = item.get("answer", "")
 
        c_col, c_bg = score_color(comp)
        l_col, l_bg = score_color(llm_s)
        sl, s_col, s_bg = sbert_label(sbert)
        f_col, f_bg = ("#059669","#d1fae5") if fills <= 2 else ("#d97706","#fef3c7") if fills <= 5 else ("#dc2626","#fee2e2")
        p_col, p_bg = ("#059669","#d1fae5") if pauses == 0 else ("#d97706","#fef3c7") if pauses <= 2 else ("#dc2626","#fee2e2")
        w_col, w_bg = ("#059669","#d1fae5") if words >= 60 else ("#d97706","#fef3c7") if words >= 25 else ("#dc2626","#fee2e2")
 
        q_cards_html += f"""
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:20px;margin-bottom:20px;">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:14px;flex-wrap:wrap;">
            <div style="background:#1e293b;color:white;font-size:12px;font-weight:700;padding:4px 12px;border-radius:6px;">Q{i+1}</div>
            {badge(f"Composite {comp}/10", c_col, c_bg)}
            {badge(f"SBERT {sbert_pct}% — {sl}", s_col, s_bg)}
            {badge(f"LLM {llm_s}/10", l_col, l_bg)}
            {badge(f"{words} words", w_col, w_bg)}
            {badge(f"Fillers: {fills}", f_col, f_bg)}
            {badge(f"Pauses: {pauses}", p_col, p_bg)}
          </div>
 
          <p style="font-size:14px;font-weight:700;color:#1e293b;margin-bottom:10px;line-height:1.5;">{q_txt}</p>
          <div style="border-left:3px solid #cbd5e1;padding:10px 14px;background:white;border-radius:0 8px 8px 0;margin-bottom:14px;">
            <p style="font-size:12px;color:#94a3b8;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;">Candidate's Answer</p>
            <p style="font-size:13px;color:#475569;font-style:italic;line-height:1.6;">"{a_txt}"</p>
          </div>
 
          <div style="margin-bottom:14px;">
            <p style="font-size:11px;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px;">Score Breakdown</p>
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
              <span style="font-size:11px;color:#94a3b8;width:90px;flex-shrink:0;">Semantic (SBERT)</span>
              <div style="flex:1;height:6px;background:#e2e8f0;border-radius:3px;"><div style="width:{sbert_pct}%;height:100%;background:{s_col};border-radius:3px;"></div></div>
              <span style="font-size:11px;font-weight:700;color:{s_col};width:52px;text-align:right;">{sbert_pct}%</span>
            </div>
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
              <span style="font-size:11px;color:#94a3b8;width:90px;flex-shrink:0;">LLM Judgment</span>
              <div style="flex:1;height:6px;background:#e2e8f0;border-radius:3px;"><div style="width:{round((llm_s/10)*100)}%;height:100%;background:{l_col};border-radius:3px;"></div></div>
              <span style="font-size:11px;font-weight:700;color:{l_col};width:52px;text-align:right;">{llm_s}/10</span>
            </div>
            <div style="display:flex;align-items:center;gap:8px;">
              <span style="font-size:11px;color:#94a3b8;width:90px;flex-shrink:0;">Composite</span>
              <div style="flex:1;height:6px;background:#e2e8f0;border-radius:3px;"><div style="width:{round((comp/10)*100)}%;height:100%;background:{c_col};border-radius:3px;"></div></div>
              <span style="font-size:11px;font-weight:700;color:{c_col};width:52px;text-align:right;">{comp}/10</span>
            </div>
          </div>
 
          <div style="background:white;border-left:4px solid {c_col};padding:12px 16px;border-radius:0 8px 8px 0;">
            <p style="font-size:12px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px;">Evaluator Feedback</p>
            <p style="font-size:13px;color:#334155;line-height:1.7;">{fb}</p>
          </div>
        </div>"""
 
    # ------------------------------------------------------------------ #
    #  8. Score trajectory                                                 #
    #                                                                      #
    #  A bar "trajectory" with 1-2 data points is visually empty and      #
    #  misleading, so we render compact score cards for short sessions   #
    #  and only use the bar chart when there are enough points for the   #
    #  shape to be informative.                                           #
    # ------------------------------------------------------------------ #
    if len(transcript) >= 3:
        # Traditional trajectory bars.
        bar_items = ""
        for i, item in enumerate(transcript):
            h = round((item["score"] / 10) * 70)
            c, _ = score_color(item["score"])
            bar_items += f"""
            <div style="display:flex;flex-direction:column;align-items:center;gap:4px;flex:1;">
              <span style="font-size:12px;font-weight:700;color:#475569;">{item['score']}</span>
              <div style="width:100%;height:{h}px;background:{c};border-radius:4px 4px 0 0;min-height:4px;"></div>
              <span style="font-size:11px;color:#94a3b8;">Q{i+1}</span>
            </div>"""
        trajectory_title = "Score Trajectory Per Question"
        trajectory_block = f"""
        <div style="display:flex;align-items:flex-end;gap:10px;height:100px;padding:0 4px;">
          {bar_items}
        </div>"""
    else:
        # Cards layout for 1-2 questions.
        card_items = ""
        for i, item in enumerate(transcript):
            c, c_bg = score_color(item["score"])
            card_items += f"""
            <div style="flex:1;background:{c_bg};border:1px solid {c}22;border-radius:10px;padding:16px 20px;display:flex;align-items:center;justify-content:space-between;">
              <div>
                <div style="font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:0.06em;">Question {i+1}</div>
                <div style="font-size:12px;color:#475569;margin-top:4px;">{item.get('word_count', 0)} words · {item.get('filler_count', 0)} fillers</div>
              </div>
              <div style="font-size:32px;font-weight:800;color:{c};line-height:1;">{item['score']}<span style="font-size:14px;color:{c}88;font-weight:600;">/10</span></div>
            </div>"""
        trajectory_title = "Per-Question Scores"
        trajectory_block = f"""
        <div style="display:flex;gap:12px;flex-wrap:wrap;">
          {card_items}
        </div>"""
 
    # ------------------------------------------------------------------ #
    #  9. Fluency per-question table                                       #
    # ------------------------------------------------------------------ #
    fluency_rows = ""
    for i, item in enumerate(transcript):
        fills  = item.get("filler_count", 0)
        pauses = item.get("long_pauses", 0)
        words  = item.get("word_count", 0)
        f_col  = "#059669" if fills <= 2 else "#d97706" if fills <= 5 else "#dc2626"
        p_col  = "#059669" if pauses == 0 else "#d97706" if pauses <= 2 else "#dc2626"
        w_col  = "#059669" if words >= 60 else "#d97706" if words >= 25 else "#dc2626"
        fluency_rows += f"""
        <tr style="border-bottom:1px solid #f1f5f9;">
          <td style="padding:10px 12px;font-weight:600;color:#1e293b;">Q{i+1}</td>
          <td style="padding:10px 12px;color:{w_col};font-weight:600;">{words}</td>
          <td style="padding:10px 12px;color:{f_col};font-weight:600;">{fills}</td>
          <td style="padding:10px 12px;color:{p_col};font-weight:600;">{pauses}</td>
          <td style="padding:10px 12px;font-weight:600;color:#1e293b;">{item['score']}/10</td>
        </tr>"""
 
    # ------------------------------------------------------------------ #
    # 10. Full HTML document                                               #
    # ------------------------------------------------------------------ #
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>MockMaster — Candidate Evaluation Report</title>
<style>
  * {{ box-sizing:border-box;margin:0;padding:0; }}
  body {{ font-family:'Segoe UI',Arial,sans-serif;background:#f1f5f9;color:#1e293b;padding:32px 20px; }}
  .page {{ max-width:880px;margin:0 auto; }}
  .section-title {{ font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;color:#94a3b8;margin:28px 0 12px;border-bottom:1px solid #e2e8f0;padding-bottom:6px;
    page-break-after:avoid;break-after:avoid; }}
  .card {{ background:white;border:1px solid #e2e8f0;border-radius:12px;padding:22px 26px;margin-bottom:14px;
    page-break-inside:avoid;break-inside:avoid; }}
  .prose {{ font-size:14px;color:#334155;line-height:1.8; }}
  .grid-4 {{ display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:14px;
    page-break-inside:avoid;break-inside:avoid; }}
  .grid-2 {{ display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin-bottom:14px;
    page-break-inside:avoid;break-inside:avoid; }}
  .metric {{ background:white;border:1px solid #e2e8f0;border-radius:10px;padding:18px;text-align:center; }}
  .metric-label {{ font-size:11px;color:#94a3b8;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px; }}
  .metric-value {{ font-size:28px;font-weight:700; }}
  table {{ width:100%;border-collapse:collapse;
    page-break-inside:avoid;break-inside:avoid; }}
  th {{ text-align:left;padding:10px 12px;font-size:11px;color:#94a3b8;text-transform:uppercase;letter-spacing:0.05em;border-bottom:2px solid #e2e8f0; }}
  tr {{ page-break-inside:avoid;break-inside:avoid; }}
  .report-footer {{ page-break-before:avoid;break-before:avoid; }}
  @media print {{ body {{ background:white;padding:0; }} .page {{ max-width:100%; }} }}
</style>
</head>
<body><div class="page">
 
  <!-- HEADER -->
  <div style="background:linear-gradient(135deg,#1e293b 0%,#0f172a 100%);color:white;border-radius:16px;padding:36px 40px;margin-bottom:24px;">
    <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:20px;">
      <div>
        <div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.12em;color:#64748b;margin-bottom:8px;">MockMaster · Candidate Evaluation Report</div>
        <div style="font-size:32px;font-weight:800;margin-bottom:6px;">Interview Deep Analysis</div>
        <div style="font-size:14px;color:#94a3b8;">{len(transcript)} Questions · Multimodal Assessment · SBERT + LLM Scoring</div>
      </div>
      <div style="text-align:right;">
        <div style="font-size:56px;font-weight:800;color:{verdict_color};line-height:1;">{avg_final}</div>
        <div style="font-size:13px;color:#64748b;margin-top:2px;">Composite Score / 10</div>
      </div>
    </div>
    <div style="margin-top:20px;background:rgba(255,255,255,0.07);border-radius:10px;padding:16px 20px;">
      <div style="font-size:15px;font-weight:700;color:{verdict_color};margin-bottom:4px;">{verdict_label}</div>
      <div style="font-size:13px;color:#94a3b8;">Methodology: 60% LLM Qualitative Judgment + 40% SBERT Semantic Similarity · Model: all-MiniLM-L6-v2</div>
    </div>
  </div>
 
  <!-- SECTION 0: JOB FIT ASSESSMENT -->
  <div class="section-title">Section 0 — Job Fit Assessment</div>
  <div class="card" style="border:2px solid {fit_ring};padding:28px;">
    <div style="display:flex;align-items:center;gap:32px;flex-wrap:wrap;">

      <!-- SVG circular gauge -->
      <div style="position:relative;width:120px;height:120px;flex-shrink:0;">
        <svg viewBox="0 0 36 36" style="width:120px;height:120px;transform:rotate(-90deg);">
          <circle cx="18" cy="18" r="15.9" fill="none" stroke="#e2e8f0" stroke-width="3.2"/>
          <circle cx="18" cy="18" r="15.9" fill="none" stroke="{fit_color}" stroke-width="3.2"
            stroke-dasharray="{fit_score} {100 - fit_score}" stroke-linecap="round"/>
        </svg>
        <div style="position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;">
          <span style="font-size:22px;font-weight:800;color:{fit_color};line-height:1;">{fit_score}%</span>
          <span style="font-size:9px;color:#94a3b8;text-transform:uppercase;letter-spacing:0.06em;margin-top:3px;">Job Fit</span>
        </div>
      </div>

      <!-- Verdict pill + analysis -->
      <div style="flex:1;min-width:220px;">
        <div style="display:inline-block;background:{fit_bg};color:{fit_color};font-size:13px;font-weight:700;
                    padding:5px 14px;border-radius:20px;margin-bottom:12px;letter-spacing:0.03em;">
          {fit_verdict}
        </div>
        <p class="prose">{fit_analysis}</p>
      </div>

    </div>

    <!-- Mini fit score bar legend -->
    <div style="margin-top:20px;padding-top:16px;border-top:1px solid #f1f5f9;">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
        <span style="font-size:12px;color:#94a3b8;width:130px;flex-shrink:0;">Role Fit Score</span>
        <div style="flex:1;height:10px;background:#e2e8f0;border-radius:5px;overflow:hidden;">
          <div style="width:{fit_score}%;height:100%;background:{fit_color};border-radius:5px;
                      transition:width 0.6s ease;"></div>
        </div>
        <span style="font-size:13px;font-weight:700;color:{fit_color};width:44px;text-align:right;">{fit_score}%</span>
      </div>
      <div style="display:flex;gap:20px;margin-top:8px;">
        <span style="font-size:11px;color:#94a3b8;">🟢 75–100% Excellent / Good Fit</span>
        <span style="font-size:11px;color:#94a3b8;">🟡 50–74% Partial Fit</span>
        <span style="font-size:11px;color:#94a3b8;">🔴 0–49% Low Fit / Mismatched Role</span>
      </div>
    </div>
  </div>

  <!-- SECTION 1: SCORE OVERVIEW -->
  <div class="section-title">Section 1 — Score Overview</div>
  <div class="grid-4">
    <div class="metric"><div class="metric-label">Composite (adjusted)</div><div class="metric-value" style="color:{verdict_color};">{avg_final}</div><div style="font-size:11px;color:#94a3b8;margin-top:4px;">technical + behavioral</div></div>
    <div class="metric"><div class="metric-label">Technical avg</div><div class="metric-value" style="color:#475569;">{avg_technical}</div><div style="font-size:11px;color:#94a3b8;margin-top:4px;">LLM + SBERT only</div></div>
    <div class="metric"><div class="metric-label">Behavioral modifier</div><div class="metric-value" style="color:{'#059669' if behavioral_modifier >= 0 else '#dc2626'};">{'+'if behavioral_modifier>=0 else ''}{behavioral_modifier}</div><div style="font-size:11px;color:#94a3b8;margin-top:4px;">body language impact</div></div>
    <div class="metric"><div class="metric-label">Semantic Match</div><div class="metric-value" style="color:#0284c7;">{round(avg_semantic*100)}%</div><div style="font-size:11px;color:#94a3b8;margin-top:4px;">SBERT cosine similarity</div></div>
  </div>
 
  <div class="card">
    <div style="font-size:13px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:16px;">{trajectory_title}</div>
    {trajectory_block}
  </div>
 
  <!-- SECTION 2: TECHNICAL ANALYSIS -->
  <div class="section-title">Section 2 — Technical Knowledge Analysis</div>
  <div class="card">
    <p class="prose">{tech_text}</p>
  </div>
 
  <!-- SECTION 3: PER-QUESTION DEEP DIVE -->
  <div class="section-title">Section 3 — Per-Question Deep Analysis</div>
  {q_cards_html}
 
  <!-- SECTION 4: FLUENCY TABLE -->
  <div class="section-title">Section 4 — Speech & Fluency Analytics</div>
  <div class="grid-2">
    <div style="background:#fef3c7;border-radius:10px;padding:20px;text-align:center;">
      <div style="font-size:40px;font-weight:800;color:#d97706;">{total_fillers}</div>
      <div style="font-size:13px;font-weight:700;color:#92400e;margin-top:4px;">Total Filler Words</div>
      <div style="font-size:11px;color:#b45309;margin-top:3px;">"um", "uh", "like", "hmm" detected</div>
    </div>
    <div style="background:#fee2e2;border-radius:10px;padding:20px;text-align:center;">
      <div style="font-size:40px;font-weight:800;color:#dc2626;">{total_pauses}</div>
      <div style="font-size:13px;font-weight:700;color:#991b1b;margin-top:4px;">Awkward Hesitations</div>
      <div style="font-size:11px;color:#b91c1c;margin-top:3px;">Silences &gt; 1.2 seconds detected</div>
    </div>
  </div>
  <div class="card" style="margin-bottom:14px;">
    <table>
      <thead><tr>
        <th>Question</th><th>Word Count</th><th>Filler Words</th><th>Hesitations</th><th>Final Score</th>
      </tr></thead>
      <tbody>{fluency_rows}</tbody>
    </table>
  </div>
  <div class="card"><p class="prose">{comm_text}</p></div>
 
  <!-- SECTION 5: BEHAVIOURAL ANALYSIS -->
  <div class="section-title">Section 5 — Behavioural Analysis & Score Contribution</div>
  <div class="card" style="margin-bottom:14px;">
    <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px;margin-bottom:18px;">
      <div>
        <div style="font-size:12px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px;">Behavioral Modifier Applied</div>
        <div style="font-size:28px;font-weight:800;color:{'#059669' if behavioral_modifier >= 0 else '#dc2626'};">{'+' if behavioral_modifier >= 0 else ''}{behavioral_modifier} pts</div>
        <div style="font-size:12px;color:#94a3b8;margin-top:2px;">Technical avg {avg_technical} → Adjusted final {avg_final}</div>
      </div>
      <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:14px 18px;text-align:center;">
        <div style="font-size:11px;color:#94a3b8;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px;">Stress — Blink Rate</div>
        <div style="font-size:16px;font-weight:700;color:{stress_color};">{stress_label}</div>
        <div style="font-size:12px;color:#64748b;margin-top:3px;">{blink_count} blinks · {blink_rate_display}</div>
        <div style="font-size:11px;color:#94a3b8;margin-top:2px;">{'≤20/min baseline · 20–35/min moderate · 35+/min high' if session_duration_sec and session_duration_sec >= 10 else '0–8/q composed · 8–15/q moderate · 15+/q high'}</div>
        <div style="font-size:12px;font-weight:700;color:{'#dc2626' if blink_penalty < 0 else '#94a3b8'};margin-top:4px;">Score effect: {'+' if blink_penalty >= 0 else ''}{blink_penalty} pts</div>
      </div>
    </div>
 
    <table style="width:100%;border-collapse:collapse;margin-bottom:16px;">
      <thead>
        <tr style="border-bottom:2px solid #e2e8f0;">
          <th style="text-align:left;padding:8px 12px;font-size:11px;color:#94a3b8;text-transform:uppercase;letter-spacing:0.05em;">Signal</th>
          <th style="text-align:left;padding:8px 12px;font-size:11px;color:#94a3b8;text-transform:uppercase;letter-spacing:0.05em;">Your Score</th>
          <th style="text-align:left;padding:8px 12px;font-size:11px;color:#94a3b8;text-transform:uppercase;letter-spacing:0.05em;">Ideal Range</th>
          <th style="text-align:left;padding:8px 12px;font-size:11px;color:#94a3b8;text-transform:uppercase;letter-spacing:0.05em;">Assessment</th>
          <th style="text-align:right;padding:8px 12px;font-size:11px;color:#94a3b8;text-transform:uppercase;letter-spacing:0.05em;">Contribution</th>
        </tr>
      </thead>
      <tbody>
        <tr style="border-bottom:1px solid #f1f5f9;">
          <td style="padding:10px 12px;font-weight:600;color:#1e293b;">Eye Contact</td>
          <td style="padding:10px 12px;font-weight:700;color:{'#059669' if eye_bonus == 1.0 else '#d97706' if eye_bonus == 0.3 else '#dc2626'};">{round(eye)}%</td>
          <td style="padding:10px 12px;font-size:12px;color:#64748b;">55–90%</td>
          <td style="padding:10px 12px;font-size:12px;color:#475569;">{eye_label}</td>
          <td style="padding:10px 12px;text-align:right;font-weight:700;color:{'#059669' if eye_bonus > 0 else '#dc2626'};">{'+' if eye_bonus >= 0 else ''}{eye_bonus}</td>
        </tr>
        <tr style="border-bottom:1px solid #f1f5f9;">
          <td style="padding:10px 12px;font-weight:600;color:#1e293b;">Head Posture</td>
          <td style="padding:10px 12px;font-weight:700;color:{'#059669' if post_bonus == 0.75 else '#d97706' if post_bonus == 0.25 else '#dc2626'};">{round(post)}%</td>
          <td style="padding:10px 12px;font-size:12px;color:#64748b;">50–95%</td>
          <td style="padding:10px 12px;font-size:12px;color:#475569;">{post_label}</td>
          <td style="padding:10px 12px;text-align:right;font-weight:700;color:{'#059669' if post_bonus > 0 else '#dc2626'};">{'+' if post_bonus >= 0 else ''}{post_bonus}</td>
        </tr>
        <tr style="border-bottom:1px solid #f1f5f9;">
          <td style="padding:10px 12px;font-weight:600;color:#1e293b;">Enthusiasm</td>
          <td style="padding:10px 12px;font-weight:700;color:{'#059669' if enth_bonus == 0.5 else '#d97706' if enth_bonus == 0.1 else '#dc2626'};">{round(enth)}%</td>
          <td style="padding:10px 12px;font-size:12px;color:#64748b;">8–35%</td>
          <td style="padding:10px 12px;font-size:12px;color:#475569;">{enth_label}</td>
          <td style="padding:10px 12px;text-align:right;font-weight:700;color:{'#059669' if enth_bonus > 0 else '#dc2626'};">{'+' if enth_bonus >= 0 else ''}{enth_bonus}</td>
        </tr>
        <tr style="border-bottom:1px solid #f1f5f9;">
          <td style="padding:10px 12px;font-weight:600;color:#1e293b;">Hand Gestures</td>
          <td style="padding:10px 12px;font-weight:700;color:{'#059669' if gest_bonus == 0.5 else '#d97706' if gest_bonus == 0.1 else '#dc2626'};">{round(gest)}%</td>
          <td style="padding:10px 12px;font-size:12px;color:#64748b;">5–40%</td>
          <td style="padding:10px 12px;font-size:12px;color:#475569;">{gest_label}</td>
          <td style="padding:10px 12px;text-align:right;font-weight:700;color:{'#059669' if gest_bonus > 0 else '#dc2626'};">{'+' if gest_bonus >= 0 else ''}{gest_bonus}</td>
        </tr>
        <tr style="border-bottom:1px solid #f1f5f9;background:#f8fafc;">
          <td style="padding:10px 12px;font-weight:600;color:#1e293b;">Blink Rate</td>
          <td style="padding:10px 12px;font-weight:700;color:{stress_color};">{blink_rate_display}</td>
          <td style="padding:10px 12px;font-size:12px;color:#64748b;">{blink_rate_ideal}</td>
          <td style="padding:10px 12px;font-size:12px;color:#475569;">{stress_label}</td>
          <td style="padding:10px 12px;text-align:right;font-weight:700;color:{'#dc2626' if blink_penalty < 0 else '#94a3b8'};">{'+' if blink_penalty >= 0 else ''}{blink_penalty}</td>
        </tr>
        <tr style="background:#f1f5f9;">
          <td colspan="4" style="padding:10px 12px;font-weight:700;color:#1e293b;">Total Behavioral Modifier</td>
          <td style="padding:10px 12px;text-align:right;font-size:15px;font-weight:800;color:{'#059669' if behavioral_modifier >= 0 else '#dc2626'};">{'+' if behavioral_modifier >= 0 else ''}{behavioral_modifier}</td>
        </tr>
      </tbody>
    </table>
    <p class="prose">{beh_text}</p>
  </div>
 
  <!-- SECTION 6: ACTION PLAN -->
  <div class="section-title">Section 6 — Prioritised Action Plan</div>
  <div class="card" style="border-left:5px solid #059669;">
    <div style="font-size:15px;font-weight:700;color:#059669;margin-bottom:14px;">What to do next — ranked by impact on composite score</div>
    <p class="prose">{plan_text}</p>
  </div>
 
  <!-- FOOTER -->
  <div class="report-footer" style="text-align:center;margin-top:24px;padding:16px 20px;font-size:11px;color:#94a3b8;border-top:1px solid #e2e8f0;">
    Generated by MockMaster · Multimodal Interview Assessment System<br>
    SBERT: all-MiniLM-L6-v2 · LLM: Llama3 · Scoring: 60% LLM + 40% SBERT · Behavioral Tracking: MediaPipe<br>
    <span style="color:{fit_color};font-weight:700;">Job Fit: {fit_score}% — {fit_verdict}</span>
  </div>
 
</div></body></html>"""
 
    return {
        "status": "success",
        "average_score": avg_final,
        "average_semantic_score": avg_semantic,
        "average_llm_score": avg_llm,
        "total_questions": len(transcript),
        "history": transcript,
        "behavioral_metrics": behavioral_data,
        "detailed_html": html
    }
 
@app.get("/video_feed")
async def video_feed():
    """Streams the live AI-processed webcam feed to the React frontend."""
    return StreamingResponse(
        generate_video_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
 
 
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)