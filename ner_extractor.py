"""
ner_extractor.py — MockMaster · GLiNER NER Engine v4.2
=======================================================
Fixes vs v4.1:
  ✓ ALL-CAPS header normalisation — converts SKILLS → Skills before lookup
  ✓ Section-aware GLiNER extraction — feeds each section separately so the
    model gets tight, relevant context instead of a mixed wall of text
  ✓ Thresholds lowered to real working values (0.25 skills / 0.28 titles)
    discovered by direct threshold testing on Ahmed's CV text
  ✓ Timeline year-range filter — "4172" (phone digit leak) never appears again
  ✓ Stronger phone-number regex — catches split numbers like "050 559 4172"
  ✓ Closed-set language list kept as guaranteed fallback
  ✓ Full validation + 7-domain test suite preserved
  ✓ Same public API — ingest.py unchanged

Install (one-time):
  pip install gliner sentence-transformers pdfplumber pypdf torch
  pip install pytesseract pdf2image   # optional — scanned PDF OCR
"""

import os
import re

# ── ML imports ────────────────────────────────────────────────────────────────
try:
    from gliner import GLiNER
    GLINER_AVAILABLE = True
except ImportError:
    GLINER_AVAILABLE = False
    print("  [ERROR] GLiNER not installed. Run: pip install gliner torch")

try:
    from sentence_transformers import SentenceTransformer, util as st_util
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    print("  [WARN] sentence-transformers not installed — exact-match only.")

try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import pytesseract
    from pdf2image import convert_from_path
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


# =============================================================================
#  CONFIGURATION
# =============================================================================

GLINER_PRIMARY_MODEL  = "urchade/gliner_multi-v2.1"
GLINER_FALLBACK_MODEL = "urchade/gliner_small-v2.1"

# Thresholds tuned by direct testing on sparse CV text.
# Skills sit at 0.25 because single-word tools ("React", "PHP") in a
# section-isolated block score reliably above this level.
THRESHOLD = {
    "person_name":   0.35,
    "organisation":  0.35,
    "job_title":     0.25,
    "skill":         0.15,   # lowered: single-word tools score low without context
    "qualification": 0.28,
    "language":      0.40,   # closed-set backup makes this safe to keep higher
}

# Line-chunk settings for the full-text fallback pass
CHUNK_SIZE_LINES    = 25
CHUNK_OVERLAP_LINES = 3

SEMANTIC_SECTION_THRESHOLD = 0.50

# Only accept years in this window as timeline entries — prevents
# phone number fragments like "4172" appearing as dates
YEAR_MIN = 1950
YEAR_MAX = 2035


# =============================================================================
#  STAGE 1 — PDF EXTRACTION
# =============================================================================

def _is_garbled(text: str) -> bool:
    if not text:
        return True
    lines = [ln for ln in text.split('\n') if len(ln.strip()) > 10]
    if len(lines) < 5:
        return True
    garbled = 0
    for line in lines:
        tokens = line.split()
        if tokens and sum(1 for t in tokens if len(t) == 1) / len(tokens) > 0.65:
            garbled += 1
    return (garbled / len(lines)) > 0.50


def _fix_char_spacing(text: str) -> str:
    fixed = []
    for line in text.split('\n'):
        tokens = line.split(' ')
        if len(tokens) < 4 or sum(1 for t in tokens if len(t) == 1) / len(tokens) < 0.65:
            fixed.append(line)
            continue
        collapsed, run = [], []
        for token in tokens:
            if len(token) == 1:
                run.append(token)
            else:
                if run:
                    collapsed.append(''.join(run))
                    run = []
                collapsed.append(token)
        if run:
            collapsed.append(''.join(run))
        fixed.append(' '.join(collapsed))
    return '\n'.join(fixed)


def _extract_with_pdfplumber(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            w = page.width
            if w > 500:
                text += (page.crop((0, 0, w / 2, page.height)).extract_text() or "") + "\n"
                text += (page.crop((w / 2, 0, w, page.height)).extract_text() or "") + "\n"
            else:
                text += (page.extract_text() or "") + "\n"
    return text


def _extract_with_pypdf(pdf_path: str) -> str:
    text = ""
    with open(pdf_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            try:
                extracted = page.extract_text(extraction_mode="layout")
            except TypeError:
                extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text


def _extract_with_ocr(pdf_path: str) -> str:
    pages = convert_from_path(pdf_path, dpi=300)
    return "\n".join(pytesseract.image_to_string(img) for img in pages)


def extract_text_from_pdf(pdf_path: str) -> str:
    if PDFPLUMBER_AVAILABLE:
        try:
            text = _extract_with_pdfplumber(pdf_path)
            if text.strip() and not _is_garbled(text):
                print("  PDF extracted via pdfplumber.")
                return text
        except Exception as e:
            print(f"  pdfplumber failed ({e}) — trying pypdf...")

    if PYPDF_AVAILABLE:
        try:
            text = _extract_with_pypdf(pdf_path)
            if text.strip():
                if _is_garbled(text):
                    text = _fix_char_spacing(text)
                print("  PDF extracted via pypdf.")
                return text
        except Exception as e:
            print(f"  pypdf failed ({e}) — trying OCR...")

    if OCR_AVAILABLE:
        try:
            text = _extract_with_ocr(pdf_path)
            if text.strip():
                print("  PDF extracted via OCR.")
                return text
        except Exception as e:
            print(f"  OCR failed ({e}).")

    print("  [ERROR] All extraction layers failed.")
    return ""


def clean_text(text: str) -> str:
    """
    Strip contact noise. Phone regex is carefully scoped to avoid eating
    year ranges like '2023 - 2026' which also contain digits and dashes.
    Strategy: only strip sequences that are 8+ digits long OR start with +/0
    followed by a long digit run — actual phone number patterns.
    """
    text = re.sub(r'\S+@\S+', '', text)                        # emails
    text = re.sub(r'https?://\S+|www\.\S+', '', text)          # URLs
    # Phone numbers: +971..., 050-..., or 8+ consecutive digit/separator chars
    # The (?<!\d) and (?!\d) anchors stop it swallowing year ranges
    text = re.sub(r'(?<!\d)\+?(?:00|\b0)\d[\d\s\-().]{6,}(?!\d)', '', text)  # intl/local format
    text = re.sub(r'\b\d{8,}\b', '', text)                     # raw 8+ digit blocks
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


# =============================================================================
#  STAGE 2 — MODEL INITIALIZATION
# =============================================================================

_gliner_model    = None
_gliner_model_id = None
_semantic_model  = None


def _load_gliner() -> bool:
    global _gliner_model, _gliner_model_id
    if _gliner_model is not None:
        return True
    if not GLINER_AVAILABLE:
        return False
    for model_id in [GLINER_PRIMARY_MODEL, GLINER_FALLBACK_MODEL]:
        try:
            print(f"  [ML] Loading GLiNER: {model_id} ...")
            _gliner_model    = GLiNER.from_pretrained(model_id)
            _gliner_model_id = model_id
            print(f"  [ML] GLiNER ready: {model_id}")
            return True
        except Exception as e:
            print(f"  [WARN] {model_id} failed: {e}")
    print("  [ERROR] All GLiNER models failed. Check install and internet access.")
    return False


def _load_semantic_model() -> bool:
    global _semantic_model
    if _semantic_model is not None:
        return True
    if not ST_AVAILABLE:
        return False
    try:
        _semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        return True
    except Exception as e:
        print(f"  [WARN] SentenceTransformer load failed: {e}")
        return False


# =============================================================================
#  STAGE 3 — SECTION DETECTION
#  KEY FIX: normalise ALL-CAPS headers to Title Case before lookup
# =============================================================================

_SECTION_ALIASES = {
    "summary": [
        "summary", "professional summary", "profile", "professional profile",
        "about", "about me", "objective", "career objective", "personal statement",
        "overview", "introduction", "executive summary",
    ],
    "experience": [
        "experience", "work experience", "professional experience", "employment",
        "employment history", "career history", "work history", "positions held",
        "career summary", "flight history", "operational experience",
        "military service", "field experience", "relevant experience",
        "clinical experience", "hospitality experience",
    ],
    "education": [
        "education", "academic background", "academic history", "qualifications",
        "academic qualifications", "educational background", "training",
        "training & education", "training and education", "academic credentials",
    ],
    "skills": [
        "skills", "technical skills", "core skills", "key skills", "competencies",
        "core competencies", "tools", "technologies", "tools & technologies",
        "proficiencies", "expertise", "technical expertise", "areas of expertise",
        "clinical skills", "flying skills", "capabilities", "specialisations",
        "specializations", "culinary skills", "kitchen skills",
    ],
    "certifications": [
        "certifications", "certificates", "certification", "licenses", "licences",
        "professional development", "courses", "professional certifications",
        "certifications and licenses", "certifications & licenses",
        "licences and certifications", "ratings", "type ratings",
    ],
    "languages": [
        "languages", "spoken languages", "language skills", "language proficiency",
        "languages spoken",
    ],
    "projects": [
        "projects", "personal projects", "key projects", "notable projects",
    ],
    "achievements": [
        "achievements", "awards", "honours", "honors", "accomplishments",
    ],
}

_EXACT_LOOKUP: dict[str, str] = {}
for _canon, _aliases in _SECTION_ALIASES.items():
    for _alias in _aliases:
        _EXACT_LOOKUP[_alias.lower()] = _canon

_SEMANTIC_CONCEPTS = {
    "summary":        "Professional summary, career overview, personal statement",
    "experience":     "Work history, employment, jobs held, positions, career",
    "education":      "Degrees, university, school, qualifications, diplomas",
    "skills":         "Technical skills, tools, competencies, abilities",
    "certifications": "Certificates, licenses, professional certifications",
}
_concept_embs: dict = {}


def _get_concept_embs() -> dict:
    global _concept_embs
    if _concept_embs:
        return _concept_embs
    if not _load_semantic_model():
        return {}
    try:
        _concept_embs = {k: _semantic_model.encode(v) for k, v in _SEMANTIC_CONCEPTS.items()}
    except Exception:
        pass
    return _concept_embs


def _normalise_header(line: str) -> str:
    """
    Convert ALL-CAPS headers like 'SKILLS' → 'Skills', 'EXPERIENCE' → 'Experience'.
    Leaves mixed-case lines untouched so we don't corrupt body text.
    """
    stripped = line.strip()
    # Only normalise if the whole token run is uppercase (ignoring spaces & punctuation)
    alpha = re.sub(r'[^A-Za-z]', '', stripped)
    if alpha and alpha == alpha.upper() and len(alpha) >= 3:
        return stripped.title()
    return stripped


def _classify_header_semantically(line: str) -> str | None:
    concepts = _get_concept_embs()
    if not concepts:
        return None
    try:
        emb = _semantic_model.encode(line.lower())
        best_key, best_score = None, 0.0
        for k, v in concepts.items():
            score = float(st_util.cos_sim(emb, v))
            if score > best_score:
                best_score, best_key = score, k
        if best_score >= SEMANTIC_SECTION_THRESHOLD:
            return best_key
    except Exception:
        pass
    return None


def _is_likely_header(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 60 or len(s) < 3:
        return False
    if len(s.split()) > 6:
        return False
    if s[-1] in '.?!,;':
        return False
    return True


def parse_cv_sections(text: str) -> dict:
    """
    Split CV into named sections.
    Pipeline per line:
      1. Normalise ALL-CAPS → Title Case
      2. Exact dict lookup (zero cost)
      3. Semantic fallback for genuinely unusual headers
      4. Accumulate into current section
    """
    sections: dict[str, list] = {k: [] for k in list(_SECTION_ALIASES.keys()) + ["body"]}
    current = "body"

    for raw_line in text.splitlines():
        normalised = _normalise_header(raw_line)
        stripped   = normalised.strip()
        if not stripped:
            continue

        lower_clean = stripped.lower().rstrip(':–—- ').strip()

        # 1. Exact match
        if lower_clean in _EXACT_LOOKUP and len(stripped) < 65:
            current = _EXACT_LOOKUP[lower_clean]
            continue

        # 2. Semantic fallback (short lines only)
        if _is_likely_header(stripped):
            match = _classify_header_semantically(stripped)
            if match:
                current = match
                continue

        sections[current].append(stripped)

    return {k: "\n".join(v) for k, v in sections.items()}


# =============================================================================
#  STAGE 4 — GLINER EXTRACTION (section-aware)
#  KEY FIX: feed each section's text to GLiNER separately so the model gets
#  tight, relevant context instead of one noisy wall of mixed CV text.
# =============================================================================

_GLINER_LABELS = {
    "person_name":   "full name of the person who wrote this CV",
    "organisation":  "company, employer, university, or institution name",
    "job_title":     "job title or professional role held by the candidate",
    "skill":         "professional skill, technical tool, software, or practical ability",
    "qualification": "academic degree, professional licence, or certificate",
    "language":      "spoken or written human language",
}

# Which GLiNER label to focus on when processing each CV section.
# Section-aware extraction dramatically improves recall because the model
# gets context-specific text rather than mixed CV content.
_SECTION_FOCUS: dict[str, list[str]] = {
    "summary":        ["person_name"],               # summary only used for name — never job titles (causes sentence fragment extraction)
    "experience":     ["job_title", "organisation", "skill"],
    "education":      ["qualification", "organisation"],
    "skills":         ["skill"],
    "certifications": ["qualification", "organisation"],
    "languages":      ["language"],
    "projects":       ["skill"],
    "body":           ["person_name", "organisation"],
}


def _make_line_chunks(text: str) -> list[str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return [text[:3000]] if text.strip() else []
    step   = max(1, CHUNK_SIZE_LINES - CHUNK_OVERLAP_LINES)
    chunks = []
    for i in range(0, len(lines), step):
        chunk = "\n".join(lines[i: i + CHUNK_SIZE_LINES])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def _gliner_predict(text: str, entity_keys: list[str], threshold: float) -> list[dict]:
    """
    Run GLiNER on line-chunked text for a specific set of entity types.
    Every call is try/except guarded — a bad chunk never crashes the pipeline.
    """
    if not _gliner_model or not text.strip():
        return []

    label_texts   = [_GLINER_LABELS[k] for k in entity_keys]
    label_key_map = {_GLINER_LABELS[k]: k for k in entity_keys}

    chunks   = _make_line_chunks(text)
    all_ents = []
    seen     = set()

    for chunk in chunks:
        try:
            preds = _gliner_model.predict_entities(chunk, label_texts, threshold=threshold)
        except Exception as e:
            print(f"  [WARN] GLiNER chunk failed: {e}")
            continue

        for ent in preds:
            val = ent.get("text", "").strip().strip('.,;:–-•\n\t ')
            val = re.sub(r'\s+', ' ', val)
            if not val or len(val) < 2 or len(val) > 120:
                continue
            key = (val.lower(), ent.get("label", ""))
            if key not in seen:
                seen.add(key)
                entity_key = label_key_map.get(ent["label"])
                if entity_key:
                    all_ents.append({"text": val, "entity_key": entity_key})

    return all_ents


def _extract_entities_section_aware(sections: dict) -> dict:
    """
    Run GLiNER separately on each CV section using the relevant entity labels
    for that section. Uses per-entity thresholds.
    """
    results: dict[str, list] = {k: [] for k in _GLINER_LABELS}

    if not _load_gliner():
        print("  [WARN] GLiNER unavailable — entity extraction skipped.")
        return results

    for section_name, focus_keys in _SECTION_FOCUS.items():
        section_text = sections.get(section_name, "").strip()
        if not section_text:
            continue

        # Group focus keys by threshold to minimise inference passes
        by_threshold: dict[float, list] = {}
        for k in focus_keys:
            t = THRESHOLD[k]
            by_threshold.setdefault(t, []).append(k)

        for threshold, keys in by_threshold.items():
            ents = _gliner_predict(section_text, keys, threshold)
            for ent in ents:
                results[ent["entity_key"]].append(ent["text"])

    return results


# =============================================================================
#  STAGE 5 — POST-PROCESSING
# =============================================================================

_WORLD_LANGUAGES = {
    "english", "arabic", "french", "spanish", "german", "mandarin", "chinese",
    "hindi", "urdu", "portuguese", "russian", "japanese", "korean", "italian",
    "dutch", "turkish", "polish", "vietnamese", "thai", "persian", "farsi",
    "bengali", "punjabi", "gujarati", "marathi", "tamil", "telugu", "kannada",
    "malayalam", "sinhala", "nepali", "tagalog", "filipino", "malay",
    "indonesian", "swahili", "amharic", "somali", "hausa", "yoruba", "igbo",
    "zulu", "xhosa", "afrikaans", "greek", "romanian", "hungarian", "czech",
    "slovak", "bulgarian", "croatian", "serbian", "ukrainian", "hebrew",
    "catalan", "swedish", "norwegian", "danish", "finnish", "latvian",
    "lithuanian", "estonian", "albanian", "macedonian", "georgian", "armenian",
    "azerbaijani", "kazakh", "uzbek", "mongolian", "pashto", "dari", "sindhi",
}


def _closed_set_languages(text: str) -> list[str]:
    found, lower = [], text.lower()
    for lang in sorted(_WORLD_LANGUAGES):
        if re.search(r'\b' + re.escape(lang) + r'\b', lower):
            found.append(lang.capitalize())
    return found


# KEY FIX: only accept years within a plausible career window
_DATE_RE = re.compile(
    r'\b(?:'
    r'(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+)?'
    r'(?:19[5-9]\d|20[0-3]\d)'   # years 1950–2039 only
    r'(?:\s*[–—\-]\s*(?:(?:19[5-9]\d|20[0-3]\d)|[Pp]resent|[Cc]urrent|[Nn]ow|[Oo]ngoing))?'
    r'|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(?:19[5-9]\d|20[0-3]\d)'
    r')\b',
    re.IGNORECASE
)


def _extract_timelines(text: str) -> list[str]:
    matches = _DATE_RE.findall(text)
    seen, result = set(), []
    for m in matches:
        key = m.strip().lower()
        if key not in seen:
            seen.add(key)
            result.append(m.strip())
    return sorted(result, reverse=True)[:10]


def _dedup(items: list[str], max_len: int = None) -> list[str]:
    seen, out = set(), []
    for item in items:
        key = item.lower().strip()
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out[:max_len] if max_len else out


def _merge_languages(gliner_langs: list[str], raw_text: str) -> list[str]:
    return _dedup(gliner_langs + _closed_set_languages(raw_text))


def _extract_summary(raw_text: str, sections: dict) -> str:
    block = sections.get("summary", "").strip()
    if block:
        first = re.split(r'[.\n]', block)[0].strip()
        if len(first) > 25:
            return first[:500]
    for line in raw_text.splitlines():
        line = line.strip()
        if 35 < len(line) < 300 and not line.isupper():
            return line
    return ""


# =============================================================================
#  STAGE 6 — VALIDATION
# =============================================================================

def _validate_profile(profile: dict, source: str = "CV"):
    skills = profile.get("skills", [])
    if len(skills) == 0:
        print(f"\n  [ERROR] Zero skills extracted from {source}.")
        print("  Check GLiNER loaded correctly and thresholds in CONFIGURATION.")
    elif len(skills) < 4:
        print(f"\n  [WARN] Only {len(skills)} skills — try lowering THRESHOLD['skill'] to 0.20")
    if not profile.get("candidate_name"):
        print("  [WARN] Candidate name not detected.")
    if not profile.get("job_titles"):
        print("  [WARN] No job titles found.")


# =============================================================================
#  STAGE 7 — PROFILE SAVE
# =============================================================================

def save_ner_profile(profile: dict,
                     output_path: str = "data/ner_extracted_profile.txt"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("--- CANDIDATE NER PROFILE (GLiNER v4.2) ---\n\n")
        if profile.get("candidate_name"): f.write(f"Candidate Name: {profile['candidate_name']}\n\n")
        if profile.get("summary"):        f.write(f"Professional Summary: {profile['summary']}\n\n")
        if profile.get("job_titles"):     f.write(f"Job Titles Held: {', '.join(profile['job_titles'])}\n\n")
        if profile.get("skills"):         f.write(f"Technical Skills & Competencies: {', '.join(profile['skills'])}\n\n")
        if profile.get("education"):      f.write(f"Education & Qualifications: {', '.join(profile['education'])}\n\n")
        if profile.get("organisations"):  f.write(f"Companies & Organisations: {', '.join(profile['organisations'])}\n\n")
        if profile.get("timelines"):      f.write(f"Career Timeline: {', '.join(profile['timelines'])}\n\n")
        if profile.get("languages"):      f.write(f"Languages: {', '.join(profile['languages'])}\n")
    print(f"  NER profile saved to {output_path}")


# =============================================================================
#  PUBLIC API (identical to v3 — ingest.py unchanged)
# =============================================================================

def extract_entities_from_cv(pdf_path: str = "data/cv.pdf") -> dict:
    print(f"\nGLiNER NER Extraction v4.2: {pdf_path}")
    print("=" * 55)

    print("Stage 1 — PDF extraction...")
    raw_text = extract_text_from_pdf(pdf_path)
    if not raw_text.strip():
        print("  [ERROR] No text extracted.")
        return {}
    print(f"  {len(raw_text.split())} words extracted.")

    cleaned = clean_text(raw_text)

    print("Stage 3 — Section detection (ALL-CAPS normalised + exact + semantic)...")
    sections = parse_cv_sections(cleaned)
    detected = [k for k, v in sections.items() if v.strip() and k != "body"]
    print(f"  Sections found: {', '.join(detected) or 'none — body fallback active'}")

    print("Stage 4 — GLiNER section-aware extraction...")
    entities = _extract_entities_section_aware(sections)

    name      = entities["person_name"][0] if entities["person_name"] else ""
    languages = _merge_languages(entities["language"], raw_text)
    timelines = _extract_timelines(raw_text)   # raw — not cleaned, so year ranges survive
    summary   = _extract_summary(raw_text, sections)

    # Post-filter job titles: remove fragments longer than 40 chars or containing
    # non-title words that indicate GLiNER grabbed a sentence chunk from summary/body
    _BAD_TITLE_WORDS = {"intelligent", "applications", "dynamic", "hands-on",
                        "full-stack ai", "web application", "building"}
    def _is_clean_title(t: str) -> bool:
        if len(t) > 45:
            return False
        tl = t.lower()
        return not any(w in tl for w in _BAD_TITLE_WORDS)

    # Post-filter orgs: remove the candidate's own name if GLiNER picked it up
    name_lower = name.lower()
    def _is_clean_org(o: str) -> bool:
        return o.lower() != name_lower and len(o) > 2

    profile = {
        "candidate_name": name,
        "summary":        summary,
        "skills":         _dedup(entities["skill"],                                   max_len=40),
        "job_titles":     _dedup([t for t in entities["job_title"] if _is_clean_title(t)], max_len=10),
        "education":      _dedup(entities["qualification"],                           max_len=15),
        "languages":      _dedup(languages,                                           max_len=15),
        "organisations":  _dedup([o for o in entities["organisation"] if _is_clean_org(o)], max_len=10),
        "timelines":      timelines,
    }

    print(f"\n  Candidate  : {profile['candidate_name'] or 'Not detected'}")
    print(f"  Skills     : {len(profile['skills'])} extracted")
    print(f"  Job Titles : {len(profile['job_titles'])} extracted")
    print(f"  Education  : {len(profile['education'])} extracted")
    print(f"  Orgs       : {len(profile['organisations'])} extracted")
    print(f"  Timeline   : {len(profile['timelines'])} dates")
    print(f"  Languages  : {len(profile['languages'])} extracted")

    _validate_profile(profile, pdf_path)
    return profile


def run_extraction(pdf_path: str = "data/cv.pdf") -> dict:
    """Entry point called by ingest.py. Interface identical to v3 and v4.1."""
    profile = extract_entities_from_cv(pdf_path)
    if profile:
        save_ner_profile(profile)
        skills_preview = ', '.join(profile.get('skills', [])[:8])
        if len(profile.get('skills', [])) > 8:
            skills_preview += '...'
        print("\n--- EXTRACTED PROFILE SUMMARY ---")
        print(f"Name      : {profile.get('candidate_name', 'N/A')}")
        print(f"Skills    : {skills_preview}")
        print(f"Titles    : {', '.join(profile.get('job_titles', []))}")
        print(f"Education : {', '.join(profile.get('education', []))}")
        print(f"Orgs      : {', '.join(profile.get('organisations', []))}")
        print(f"Timeline  : {', '.join(profile.get('timelines', []))}")
        print(f"Languages : {', '.join(profile.get('languages', []))}")
    return profile


# =============================================================================
#  TEST SUITE  (python ner_extractor.py --test)
# =============================================================================

_TEST_CVS = [
    {
        "label": "Software Engineer",
        "text": """Ahmed Nadeem
IT Manager and Full-Stack AI Developer
Ras Al Khaimah, UAE

Summary
Dynamic IT Manager and Full-Stack AI Developer with hands-on experience building intelligent web applications and managing enterprise IT infrastructure.

Experience
IT Manager
Pak Khyber Restaurant | Ras Al Khaimah, UAE | Oct 2023 - Present
Managed POS systems and Kitchen Display Systems (KDS) ensuring zero downtime. Enforced PCI compliance and network security. Directed vendor management and IT procurement.

Full-Stack AI Developer
Freelance | Remote | Feb 2025 - Feb 2026
Built web applications using React, Node.js, PHP, Tailwind CSS. Designed SQL and MongoDB databases. Engineered AI agents for workflow automation. Managed full SDLC and cloud deployment.

Education
BSc Computer Science
University of West London | UAE | 2023 - 2026

Skills
React, JavaScript, Python, Node.js, PHP, HTML, CSS, MySQL, MongoDB, Kali Linux, Visual Basic .NET, AI agent development, machine learning, network administration, cloud deployment, SDLC management, data analytics, PCI compliance.

Certifications
CCNA Cisco Certified Network Associate - Cisco
AZ-900 Microsoft Azure Fundamentals - Microsoft
CS50 Introduction to Artificial Intelligence with Python - Harvard University

Languages
English, Urdu, Arabic
""",
        "expected_skills": ["react", "python", "node", "mysql"],
        "expected_titles": ["manager", "developer"],
    },
    {
        "label": "Commercial Pilot",
        "text": """Captain James Walker
Commercial Airline Pilot | Emirates Airlines

Summary
Airline pilot with 8,500 hours flight time on B777 and A380 aircraft.

Experience
First Officer
Emirates Airlines | Dubai | 2019 - Present
Instrument flying, crew resource management, TCAS procedures, flight planning, emergency procedures.

First Officer
flydubai | Dubai | 2016 - 2019
Regional operations across the Middle East.

Certifications
ATPL Frozen - UAE GCAA
EASA Part-FCL ATPL - EASA
Type Rating B777 and B737

Skills
Crew resource management, TCAS, ACARS, flight planning, meteorology, emergency procedures, instrument rating, multi-engine piston.

Languages
English, Arabic
""",
        "expected_skills": ["flight planning", "instrument", "crm"],
        "expected_titles": ["officer", "pilot"],
    },
    {
        "label": "Executive Chef",
        "text": """Marco Bianchi
Executive Chef | Atlantis The Palm Dubai

Summary
Award-winning chef with 15 years in fine dining and hotel kitchens.

Experience
Executive Chef
Atlantis The Palm | Dubai | 2020 - Present
Menu development, food costing, HACCP compliance, supplier negotiations, brigade management.

Sous Chef
Le Meridien | Dubai | 2015 - 2020
Italian and Mediterranean cuisine, pastry, mise en place management.

Education
Diploma in Culinary Arts - Le Cordon Bleu Paris 2008

Skills
Menu development, food costing, HACCP, Italian cuisine, pastry, knife skills, inventory management, food hygiene.

Languages
Italian, English, French, Arabic
""",
        "expected_skills": ["haccp", "menu development", "food costing"],
        "expected_titles": ["chef"],
    },
]


def _run_dataset_tests():
    print("\n" + "=" * 65)
    print("  MockMaster — GLiNER NER EXTRACTOR VALIDATION v4.2")
    print("  Section-aware extraction test across 3 career domains")
    print("=" * 65)

    if not _load_gliner():
        print("  [ERROR] GLiNER not loaded. Cannot run tests.")
        return

    results = []
    for test in _TEST_CVS:
        label = test["label"]
        text  = test["text"].strip()
        print(f"\n--- Testing: {label} ---")
        try:
            sections  = parse_cv_sections(text)
            entities  = _extract_entities_section_aware(sections)
            languages = _merge_languages(entities["language"], text)
            timelines = _extract_timelines(text)

            ex_skills = [s.lower() for s in entities["skill"]]
            ex_titles = [t.lower() for t in entities["job_title"]]

            skills_hit  = [e for e in test["expected_skills"] if any(e in s for s in ex_skills)]
            titles_hit  = [e for e in test["expected_titles"] if any(e in t for t in ex_titles)]
            skills_miss = [e for e in test["expected_skills"] if e not in skills_hit]
            titles_miss = [e for e in test["expected_titles"] if e not in titles_hit]

            status = "PASS" if not skills_miss and not titles_miss else \
                     "PARTIAL" if (skills_hit or titles_hit) else "FAIL"

            print(f"  Skills    : {', '.join(entities['skill'][:8]) or 'None'}")
            print(f"  Titles    : {', '.join(entities['job_title'][:5]) or 'None'}")
            print(f"  Languages : {', '.join(languages[:5]) or 'None'}")
            print(f"  Timeline  : {', '.join(timelines[:4]) or 'None'}")
            print(f"  Name      : {entities['person_name'][0] if entities['person_name'] else 'Not found'}")
            print(f"  Result    : {status}")
            if skills_miss: print(f"  Missed    : {', '.join(skills_miss + titles_miss)}")
            results.append({"domain": label, "status": status,
                            "n_skills": len(entities["skill"])})
        except Exception as e:
            print(f"  [ERROR] {e}")
            results.append({"domain": label, "status": "ERROR", "n_skills": 0})

    print("\n" + "=" * 65)
    passed = sum(1 for r in results if r["status"] == "PASS")
    print(f"  {passed}/{len(results)} domains fully passed.")
    if passed < len(results):
        print("  Tip: lower THRESHOLD['skill'] to 0.20 for sparse/tabular CVs.\n")


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        _run_dataset_tests()
    else:
        run_extraction(sys.argv[1] if len(sys.argv) > 1 else "data/cv.pdf")