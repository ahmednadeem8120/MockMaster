import React, { useState, useRef, useEffect } from 'react';
import { Mic, Square, PlayCircle, Activity, Clock, Target, Zap, BarChart3, ArrowRight, UploadCloud, FileText, CheckCircle, Download, Sparkles } from 'lucide-react';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip } from 'recharts';
import html2pdf from 'html2pdf.js';

const API_BASE = "http://localhost:8000";

// --- FLOATING TECH BADGES (like the PrepInterview screenshot) ---
const TECH_BADGES = [
  { label: 'Whisper ASR', icon: '🎙️', color: 'from-blue-50 to-blue-100', text: 'text-blue-700', border: 'border-blue-200' },
  { label: 'SBERT Semantic', icon: '🧠', color: 'from-emerald-50 to-emerald-100', text: 'text-emerald-700', border: 'border-emerald-200' },
  { label: 'MediaPipe Vision', icon: '👁️', color: 'from-orange-50 to-orange-100', text: 'text-orange-700', border: 'border-orange-200' },
  { label: 'Llama3 LLM', icon: '⚡', color: 'from-purple-50 to-purple-100', text: 'text-purple-700', border: 'border-purple-200' },
  { label: 'FAISS Vector DB', icon: '🗄️', color: 'from-rose-50 to-rose-100', text: 'text-rose-700', border: 'border-rose-200' },
  { label: 'GLiNer NER', icon: '📝', color: 'from-amber-50 to-amber-100', text: 'text-amber-700', border: 'border-amber-200' },
  { label: 'Behavioral Analysis', icon: '🎯', color: 'from-teal-50 to-teal-100', text: 'text-teal-700', border: 'border-teal-200' },
];

// --- Floating Tech Badges Scroll Component ---
const FloatingTechBadges = () => {
  return (
    <div className="relative w-full overflow-hidden py-4">
      <div className="flex gap-3 animate-scroll-horizontal whitespace-nowrap">
        {[...TECH_BADGES, ...TECH_BADGES, ...TECH_BADGES].map((tech, i) => (
          <div
            key={i}
            className={`flex items-center gap-2 px-5 py-3 rounded-full bg-gradient-to-br ${tech.color} border ${tech.border} shadow-sm hover:shadow-md transition-all duration-300 flex-shrink-0`}
          >
            <span className="text-lg">{tech.icon}</span>
            <span className={`text-sm font-semibold ${tech.text}`}>{tech.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default function App() {
  const [status, setStatus] = useState('setup'); 
  const [currentQuestion, setCurrentQuestion] = useState("System ready. Initialize to commence assessment.");
  const [latestFeedback, setLatestFeedback] = useState("");
  const [transcript, setTranscript] = useState([]);
  const [debriefData, setDebriefData] = useState(null);
  const [recordingTime, setRecordingTime] = useState(0);
  
  const [cvFile, setCvFile] = useState(null);
  const [jdFile, setJdFile] = useState(null);
  const [jdText, setJdText] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [difficulty, setDifficulty] = useState('Medium');
  const [numQuestions, setNumQuestions] = useState(5);
  const [isInterviewComplete, setIsInterviewComplete] = useState(false);
  
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const canvasRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const animationFrameRef = useRef(null);

  useEffect(() => {
    let interval;
    if (status === 'listening') {
      interval = setInterval(() => setRecordingTime(prev => prev + 1), 1000);
    } else {
      setRecordingTime(0);
      clearInterval(interval);
    }
    return () => clearInterval(interval);
  }, [status]);

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs < 10 ? '0' : ''}${secs}`;
  };

  const getStressLevel = (blinks, sessionDurationSec) => {
    const mins = Math.max(0.1, (sessionDurationSec || 0) / 60);
    const blinksPerMin = Math.round((blinks / mins) * 10) / 10;
    if (blinksPerMin > 35) return { 
        label: "High Stress", 
        color: "text-red-600", 
        bg: "bg-red-50",
        detail: `${blinksPerMin}/min`
    };
    if (blinksPerMin > 20) return { 
        label: "Moderate Stress", 
        color: "text-amber-600", 
        bg: "bg-amber-50",
        detail: `${blinksPerMin}/min`
    };
    return { 
        label: "Composed and Calm", 
        color: "text-emerald-600", 
        bg: "bg-emerald-50",
        detail: `${blinksPerMin}/min`
    };
};

  const speak = (text) => {
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.95;
    window.speechSynthesis.speak(utterance);
  };

  const drawWaveform = () => {
    if (!canvasRef.current || !analyserRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const analyser = analyserRef.current;
    
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    const draw = () => {
      animationFrameRef.current = requestAnimationFrame(draw);
      analyser.getByteTimeDomainData(dataArray);
      
      ctx.fillStyle = 'rgba(255, 247, 237, 0.3)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      ctx.lineWidth = 3;
      ctx.strokeStyle = '#f97316'; // orange accent matching the theme
      ctx.beginPath();
      
      const sliceWidth = canvas.width * 1.0 / bufferLength;
      let x = 0;
      
      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = v * canvas.height / 2;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
        x += sliceWidth;
      }
      ctx.lineTo(canvas.width, canvas.height / 2);
      ctx.stroke();
    };
    draw();
  };

  // fetch() with a configurable timeout. If the backend hangs (e.g. Ollama
  // is offline or the FAISS build stalls), the request will abort and the
  // user sees a real error instead of an infinite spinner.
  const fetchWithTimeout = async (url, options = {}, timeoutMs = 15000) => {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const res = await fetch(url, { ...options, signal: controller.signal });
      return res;
    } finally {
      clearTimeout(timeoutId);
    }
  };

  const handleUpload = async () => {
    if (!cvFile) return alert("Please upload your CV.");
    if (!jdFile && !jdText.trim()) return alert("Please either upload a Job Description file or paste the text.");
    setIsUploading(true);
    setUploadError("");
    setStatus('processing');

    const formData = new FormData();
    formData.append("cv", cvFile);

    // If the user pasted text, wrap it in a Blob so the backend's UploadFile
    // handler gets the same shape it already expects. No backend changes.
    if (jdFile) {
      formData.append("jd", jdFile);
    } else {
      const jdBlob = new Blob([jdText], { type: "text/plain" });
      formData.append("jd", jdBlob, "pasted_job_description.txt");
    }

    try {
      // Knowledge base build includes FAISS embedding + NER extraction,
      // so the timeout is generous (120s) but bounded.
      const res = await fetchWithTimeout(
        `${API_BASE}/upload`,
        { method: "POST", body: formData },
        120000
      );
      if (!res.ok) throw new Error(`Server responded with ${res.status}`);

      setStatus('idle');
      setCurrentQuestion("Knowledge base synchronized. Configure settings and initialize assessment.");
    } catch (err) {
      const msg = err.name === "AbortError"
        ? "Upload timed out after 2 minutes. Check that the backend (FastAPI + Ollama) is running."
        : `Could not reach the backend: ${err.message}. Verify the server is running at ${API_BASE}.`;
      setUploadError(msg);
      setStatus('setup');
    } finally {
      setIsUploading(false);
    }
  };

  const startInterview = async () => {
    setStatus('processing');
    setUploadError("");
    try {
      const response = await fetchWithTimeout(
        `${API_BASE}/start?difficulty=${difficulty}&questions=${numQuestions}`,
        {},
        60000
      );
      if (!response.ok) throw new Error(`Server responded with ${response.status}`);
      const data = await response.json();
      setCurrentQuestion(data.question);
      setLatestFeedback("");
      speak(data.question);
      setStatus('asking');
    } catch (error) {
      const msg = error.name === "AbortError"
        ? "Question generation timed out. Check that Ollama is running and responsive."
        : `Connection error: ${error.message}`;
      setCurrentQuestion(msg);
      setStatus('idle');
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      audioContextRef.current = new AudioContext();
      analyserRef.current = audioContextRef.current.createAnalyser();
      const source = audioContextRef.current.createMediaStreamSource(stream);
      source.connect(analyserRef.current);
      analyserRef.current.fftSize = 256;
      
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
      };

      mediaRecorderRef.current.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        mediaRecorderRef.current?.stream?.getTracks().forEach(t => t.stop());
        if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
        if (audioContextRef.current) audioContextRef.current.close();
        
        await sendAnswerToAPI(audioBlob);
      };

      mediaRecorderRef.current.start();
      setStatus('listening');
      setTimeout(drawWaveform, 100);

    } catch (err) {
      alert("Microphone access is required.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
      setStatus('processing');
    }
  };

  const sendAnswerToAPI = async (audioBlob) => {
    const formData = new FormData();
    formData.append("audio", audioBlob, "answer.webm");

    try {
      // /reply runs Whisper transcription + SBERT + an LLM round-trip,
      // which can legitimately take 30-60s on CPU, so the timeout is 90s.
      const response = await fetchWithTimeout(
        `${API_BASE}/reply`,
        { method: "POST", body: formData },
        90000
      );
      if (!response.ok) throw new Error(`Server crashed with status: ${response.status}`);
      
      const data = await response.json();
      
      const semanticPct = Math.round((data.semantic_score || 0) * 100);
      setLatestFeedback(`Final: ${data.score}/10 · Semantic Match: ${semanticPct}% · LLM: ${data.llm_score}/10 — ${data.feedback}`);
      setCurrentQuestion(data.next_question);
      setTranscript(prev => [...prev, {
        q: currentQuestion,
        a: data.transcription,
        score: data.score,
        llm_score: data.llm_score,
        semantic_score: data.semantic_score
      }]);
      
      if (data.is_complete) {
        setIsInterviewComplete(true);
      }
      
      speak(data.next_question); 
      setStatus('asking');
    } catch (error) {
      const msg = error.name === "AbortError"
        ? "Answer processing timed out. Whisper or Ollama may be overloaded — please try a shorter answer."
        : "System Warning: AI Output parsing failed. Please try answering again.";
      setLatestFeedback(msg);
      setCurrentQuestion("Could you repeat that last point or elaborate further?");
      speak("I encountered an error processing your answer. Could you elaborate further?");
      setStatus('asking');
    }
  };

  const endInterview = async () => {
    setStatus('processing');
    try {
      // /end runs a second LLM call to generate the full prose report, plus
      // waits for the behavioral tracker to flush its final metrics, so the
      // timeout is generous (120s).
      const response = await fetchWithTimeout(`${API_BASE}/end`, {}, 120000);
      if (!response.ok) throw new Error(`Server responded with ${response.status}`);
      const data = await response.json();
      setDebriefData(data);
      setStatus('debrief');
      speak("Session terminated. Compiling your performance analytics.");
    } catch (error) {
      const msg = error.name === "AbortError"
        ? "Report generation timed out. The LLM is taking too long — please try again."
        : `Could not generate report: ${error.message}`;
      setLatestFeedback(msg);
      setStatus('idle');
    }
  };

  const downloadPDF = () => {
    const element = document.getElementById('printable-report');
    element.style.display = 'block';

    const opt = {
      margin:       [0.4, 0.4, 0.5, 0.4],   // top, right, bottom, left (inches)
      filename:     'MockMaster_AI_Deep_Analysis.pdf',
      image:        { type: 'jpeg', quality: 0.98 },
      html2canvas:  { scale: 2, useCORS: true, letterRendering: true },
      jsPDF:        { unit: 'in', format: 'letter', orientation: 'portrait', compress: true },
      // Prevent elements from being split mid-block across pages — stops the
      // common failure mode where only a footer ends up on the final page.
      pagebreak:    { mode: ['avoid-all', 'css', 'legacy'] }
    };

    html2pdf().set(opt).from(element).save().then(() => {
      element.style.display = 'none';
    });
  };

  // --- DEBRIEF DASHBOARD (LIGHT THEME) ---
  if (status === 'debrief' && debriefData) {
    const chartData = debriefData.history.map((item, index) => ({
      name: `Q${index + 1}`,
      score: item.score
    }));
    const totalFillers = debriefData.history.reduce((sum, item) => sum + (item.filler_count || 0), 0);
    const totalPauses = debriefData.history.reduce((sum, item) => sum + (item.long_pauses || 0), 0);

    return (
      <div className="min-h-screen bg-gradient-to-br from-orange-50 via-rose-50 to-amber-50 text-slate-800 p-8 flex flex-col items-center">
        
        <div className="w-full max-w-5xl flex justify-end mb-4 z-10 relative">
          <button onClick={downloadPDF} className="flex items-center gap-2 bg-orange-500 hover:bg-orange-600 text-white px-6 py-3 rounded-full font-bold transition-all shadow-lg shadow-orange-500/25">
            <Download className="w-5 h-5" /> Export Deep Analysis PDF
          </button>
        </div>

        {/* Hidden printable — the backend HTML report already contains its own
            header, composite score, and all sections. We render it as-is. */}
        <div id="printable-report" style={{ display: 'none', backgroundColor: 'white', color: 'black', fontFamily: 'Arial, sans-serif' }}>
            <div
                dangerouslySetInnerHTML={{ __html: debriefData.detailed_html }}
            />
        </div>

        <div className="w-full max-w-5xl bg-white/60 backdrop-blur-xl p-8 rounded-3xl shadow-xl border border-white relative z-0">

          <header className="mb-10 text-center">
            <h1 className="text-4xl font-black text-transparent bg-clip-text bg-gradient-to-r from-orange-500 to-rose-500 mb-2">Post-Interview Debrief</h1>
            <p className="text-slate-500 mb-6">Your complete performance analytics</p>
            <div className="flex justify-center gap-8 mt-3">
              <div className="text-center">
                <p className="text-slate-500 text-xs uppercase tracking-widest">Composite</p>
                <p className="text-slate-900 font-mono text-2xl font-bold">{debriefData.average_score}<span className="text-slate-400">/10</span></p>
              </div>
              <div className="text-center">
                <p className="text-slate-500 text-xs uppercase tracking-widest">Semantic Match</p>
                <p className="text-orange-600 font-mono text-2xl font-bold">{Math.round((debriefData.average_semantic_score || 0) * 100)}<span className="text-slate-400">%</span></p>
              </div>
              <div className="text-center">
                <p className="text-slate-500 text-xs uppercase tracking-widest">LLM Judgment</p>
                <p className="text-rose-600 font-mono text-2xl font-bold">{debriefData.average_llm_score}<span className="text-slate-400">/10</span></p>
              </div>
            </div>
          </header>

          {/* Behavioral Metrics */}
          <h3 className="text-lg font-bold text-slate-900 mb-4 border-b border-slate-200 pb-2">🎭 Multi-Modal Behavioral Analysis</h3>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-8">
            <div className="bg-white border border-slate-200 rounded-2xl p-4 text-center shadow-sm hover:shadow-md transition-all">
                <p className="text-slate-500 text-[10px] tracking-widest uppercase mb-2">Eye Contact</p>
                <p className="text-2xl font-bold text-emerald-600">{debriefData.behavioral_metrics?.eye_contact_score || 0}%</p>
            </div>
            <div className="bg-white border border-slate-200 rounded-2xl p-4 text-center shadow-sm hover:shadow-md transition-all">
                <p className="text-slate-500 text-[10px] tracking-widest uppercase mb-2">Head Posture</p>
                <p className="text-2xl font-bold text-blue-600">{debriefData.behavioral_metrics?.head_posture_score || 0}%</p>
            </div>
            <div className="bg-white border border-slate-200 rounded-2xl p-4 text-center shadow-sm hover:shadow-md transition-all">
                <p className="text-slate-500 text-[10px] tracking-widest uppercase mb-2">Enthusiasm</p>
                <p className="text-2xl font-bold text-purple-600">{debriefData.behavioral_metrics?.enthusiasm_score || 0}%</p>
            </div>
            <div className="bg-white border border-slate-200 rounded-2xl p-4 text-center shadow-sm hover:shadow-md transition-all">
                <p className="text-slate-500 text-[10px] tracking-widest uppercase mb-2">Hand Gestures</p>
                <p className="text-2xl font-bold text-amber-600">{debriefData.behavioral_metrics?.hand_gesture_score || 0}%</p>
            </div>
            <div className={`border border-slate-200 rounded-2xl p-4 text-center shadow-sm transition-all ${getStressLevel(debriefData.behavioral_metrics?.blink_count, debriefData.behavioral_metrics?.session_duration_sec).bg}`}>
              <p className="text-slate-500 text-[10px] tracking-widest uppercase mb-2">Stress Level</p>
              <p className={`text-xl font-black uppercase ${getStressLevel(debriefData.behavioral_metrics?.blink_count, debriefData.behavioral_metrics?.session_duration_sec).color}`}>
               {getStressLevel(debriefData.behavioral_metrics?.blink_count, debriefData.behavioral_metrics?.session_duration_sec).label}
               </p>
               <p className="text-[9px] text-slate-500 mt-1">
                {debriefData.behavioral_metrics?.blink_count || 0} blinks · {getStressLevel(debriefData.behavioral_metrics?.blink_count, debriefData.behavioral_metrics?.session_duration_sec).detail} · baseline 12–20/min
               </p>
             </div>
          </div>

          {/* Speech & Fluency */}
          <h3 className="text-lg font-bold text-slate-900 mb-4 border-b border-slate-200 pb-2 mt-8">🗣️ Speech & Vocal Fluency Analytics</h3>
          <div className="grid grid-cols-2 gap-4 mb-8">
            <div className="bg-white border border-slate-200 rounded-2xl p-6 text-center shadow-sm hover:shadow-md transition-all">
                <p className="text-slate-500 text-sm tracking-widest uppercase mb-2">Filler Words Used</p>
                <p className="text-4xl font-black text-amber-600">{totalFillers}</p>
                <p className="text-xs text-slate-500 mt-2 font-mono">Detected: "um", "uh", "like"</p>
            </div>
            <div className="bg-white border border-slate-200 rounded-2xl p-6 text-center shadow-sm hover:shadow-md transition-all">
                <p className="text-slate-500 text-sm tracking-widest uppercase mb-2">Awkward Hesitations</p>
                <p className="text-4xl font-black text-red-500">{totalPauses}</p>
                <p className="text-xs text-slate-500 mt-2 font-mono">Silences &gt; 1.2 seconds</p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
            <div className="bg-white border border-slate-200 rounded-3xl p-6 shadow-sm h-[300px]">
              <h3 className="text-sm text-slate-500 uppercase tracking-widest mb-4 flex items-center gap-2"><BarChart3 className="w-4 h-4"/> Performance Trajectory</h3>
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={chartData}>
                  <defs>
                    <linearGradient id="colorScore" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#f97316" stopOpacity={0.4}/>
                      <stop offset="95%" stopColor="#f97316" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="name" stroke="#94a3b8" />
                  <YAxis domain={[0, 10]} stroke="#94a3b8" />
                  <Tooltip contentStyle={{ backgroundColor: '#ffffff', borderColor: '#e2e8f0', borderRadius: '12px' }} />
                  <Area type="monotone" dataKey="score" stroke="#f97316" strokeWidth={3} fillOpacity={1} fill="url(#colorScore)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-gradient-to-br from-orange-400 to-rose-400 rounded-3xl p-8 shadow-lg flex flex-col justify-center items-center text-center text-white">
              <Target className="w-12 h-12 mb-4" />
              <h2 className="text-5xl font-black mb-2">{debriefData.total_questions}</h2>
              <p className="uppercase tracking-widest text-sm font-semibold opacity-90">Total Questions Analyzed</p>
            </div>
          </div>

          <div className="bg-white border border-slate-200 rounded-3xl p-8 shadow-sm">
            <h3 className="text-lg font-bold text-slate-900 mb-6 border-b border-slate-200 pb-4">Detailed Action Log</h3>
            <div className="space-y-6">
              {debriefData.history.map((item, i) => (
                <div key={i} className="flex gap-4">
                  <div className="flex flex-col items-center">
                    <div className={`w-10 h-10 rounded-full flex items-center justify-center font-bold text-sm ${item.score >= 7 ? 'bg-emerald-100 text-emerald-700' : 'bg-amber-100 text-amber-700'}`}>
                      {item.score}
                    </div>
                    <div className="w-[1px] h-full bg-slate-200 my-2"></div>
                  </div>
                  <div className="pb-6">
                    <p className="text-slate-800 font-medium mb-1">{item.question}</p>
                    <p className="text-slate-500 text-sm italic mb-2">"{item.answer}"</p>
                    <div className="flex flex-wrap gap-3 mb-3">
                      <span className="text-[10px] font-bold uppercase tracking-wider bg-orange-50 text-orange-700 px-2 py-1 rounded border border-orange-200">
                        Semantic: {Math.round((item.semantic_score || 0) * 100)}%
                      </span>
                      <span className="text-[10px] font-bold uppercase tracking-wider bg-rose-50 text-rose-700 px-2 py-1 rounded border border-rose-200">
                        LLM: {item.llm_score || 0}/10
                      </span>
                      <span className="text-[10px] font-bold uppercase tracking-wider bg-amber-50 text-amber-700 px-2 py-1 rounded border border-amber-200">
                        Fillers: {item.filler_count || 0}
                      </span>
                      <span className="text-[10px] font-bold uppercase tracking-wider bg-red-50 text-red-700 px-2 py-1 rounded border border-red-200">
                        Hesitations: {item.long_pauses || 0}
                      </span>
                    </div>
                    <p className="text-orange-600 text-sm flex items-center gap-2"><ArrowRight className="w-3 h-3"/> {item.feedback}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    );
  }

  // --- MAIN INTERVIEW VIEW (LIGHT THEME) ---
  return (
    <div className="min-h-screen bg-gradient-to-br from-orange-50 via-rose-50 to-amber-50 text-slate-800 font-sans relative overflow-hidden">
      {/* Decorative glow orbs */}
      <div className="absolute top-[-10%] right-[-5%] w-[40%] h-[40%] bg-gradient-to-br from-orange-300/40 to-rose-300/40 blur-[120px] rounded-full pointer-events-none"></div>
      <div className="absolute bottom-[-10%] left-[-5%] w-[40%] h-[40%] bg-gradient-to-br from-amber-300/30 to-orange-300/30 blur-[120px] rounded-full pointer-events-none"></div>
      
      <div className="relative z-10 p-4 md:p-8 flex flex-col items-center">
        <div className="w-full max-w-7xl flex flex-col">
          
          {/* HEADER */}
          <header className="flex justify-between items-center mb-6 bg-white/70 backdrop-blur-xl border border-white shadow-sm px-8 py-4 rounded-3xl">
            <div className="flex items-center gap-4">
              <div className="p-3 bg-gradient-to-br from-orange-400 to-rose-400 rounded-xl shadow-md shadow-orange-500/20">
                <Zap className="text-white w-6 h-6" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-slate-900 tracking-wide">MockMaster</h1>
                <p className="text-xs text-slate-500">AI-Driven Multimodal Mock Interview Platform</p>
              </div>
            </div>
            
            <div className="flex items-center gap-3">
              {status === 'listening' && (
                <div className="flex items-center gap-2 text-emerald-700 bg-emerald-50 px-4 py-2 rounded-full border border-emerald-200">
                  <Clock className="w-4 h-4 animate-pulse" />
                  <span className="font-mono font-bold tracking-wider">{formatTime(recordingTime)}</span>
                </div>
              )}
              <div className={`px-5 py-2.5 rounded-full border flex items-center gap-2 text-xs font-bold uppercase tracking-widest ${status === 'listening' ? 'bg-emerald-50 border-emerald-200 text-emerald-700' : 'bg-orange-50 border-orange-200 text-orange-700'}`}>
                <Activity className={`w-4 h-4 ${status === 'processing' ? 'animate-spin' : ''}`} />
                {status}
              </div>
            </div>
          </header>

          {/* HERO BANNER (only on setup) */}
          {status === 'setup' && (
            <div className="mb-8">
              <div className="bg-gradient-to-br from-orange-100 via-rose-100 to-amber-100 rounded-[2.5rem] p-12 relative overflow-hidden border border-white shadow-sm">
                {/* AI Techy Morphing Bubbles - Fast & Vigorous */}
                <style>
                  {`
                    @keyframes blob-morph {
                      0%, 100% { border-radius: 70% 30% 40% 60% / 60% 40% 70% 30%; }
                      50% { border-radius: 20% 80% 30% 70% / 80% 30% 40% 60%; }
                    }
                    @keyframes blob-float-outer {
                      0%, 100% { transform: translateY(-50%) scale(1) rotate(0deg); }
                      50% { transform: translateY(-65%) scale(1.15) rotate(25deg); }
                    }
                    @keyframes blob-float-inner {
                      0%, 100% { transform: translateY(-50%) scale(1) rotate(0deg); }
                      50% { transform: translateY(-30%) scale(0.85) rotate(-30deg); }
                    }
                  `}
                </style>
                <div 
                  className="absolute right-12 top-1/2 w-48 h-48 bg-gradient-to-br from-purple-300/60 via-pink-300/60 to-orange-300/60 blur-2xl hidden md:block origin-center"
                  style={{ animation: 'blob-float-outer 4s ease-in-out infinite, blob-morph 5s ease-in-out infinite' }}
                ></div>
                <div 
                  className="absolute right-20 top-1/2 w-32 h-32 bg-gradient-to-br from-white/90 to-purple-200/80 shadow-2xl hidden md:block origin-center backdrop-blur-sm border border-white/40"
                  style={{ animation: 'blob-float-inner 3.5s ease-in-out infinite, blob-morph 4s ease-in-out infinite reverse' }}
                ></div>
                
                <div className="relative z-10 max-w-2xl">
                  <div className="inline-flex items-center gap-2 bg-white/80 backdrop-blur px-4 py-2 rounded-full border border-orange-200 mb-6">
                    <Sparkles className="w-4 h-4 text-orange-500" />
                    <span className="text-xs font-semibold text-orange-700">Multimodal AI Assessment System</span>
                  </div>
                  <h1 className="text-4xl md:text-5xl font-black text-slate-900 mb-3 leading-tight">
                    Get Your Dream Job
                  </h1>
                  <h2 className="text-3xl md:text-4xl font-black text-slate-900 mb-4 leading-tight">
                    without Interview Anxiety
                  </h2>
                  <p className="text-slate-600 text-lg max-w-md">
                    AI practice interviews with behavioral analysis, semantic scoring, and personalized feedback.
                  </p>
                </div>
              </div>

              {/* FLOATING TECH BADGES — the scrolling row you wanted */}
              <div className="mt-6 bg-white/60 backdrop-blur-xl border border-white rounded-full py-2 px-2 shadow-sm">
                <FloatingTechBadges />
              </div>
            </div>
          )}

          {/* MAIN GRID */}
          <main className="grid grid-cols-1 lg:grid-cols-12 gap-6 flex-1">
            <div className="col-span-1 lg:col-span-8 flex flex-col gap-6">
              
              {status === 'setup' ? (
                <div className="min-h-[400px] relative overflow-hidden rounded-[2.5rem] bg-white/80 backdrop-blur-xl border border-white shadow-sm p-10 flex flex-col items-center justify-center">
                  <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-orange-400 to-rose-400 flex items-center justify-center mb-6 shadow-lg shadow-orange-500/20">
                    <UploadCloud className="w-8 h-8 text-white" />
                  </div>
                  <h2 className="text-3xl font-bold text-slate-900 mb-2">Synchronize Your Data</h2>
                  <p className="text-slate-500 mb-8 text-center max-w-md">Upload your CV and paste the Job Description to initialize the AI context.</p>

                  <div className="w-full max-w-2xl mb-6 space-y-4">
                    {/* CV upload tile — unchanged */}
                    <div className="bg-white border border-slate-200 rounded-2xl p-6 flex flex-col items-center justify-center hover:border-orange-300 hover:bg-orange-50/50 transition-colors cursor-pointer" onClick={() => document.getElementById('cv-upload').click()}>
                      <FileText className={`w-10 h-10 mb-3 ${cvFile ? 'text-emerald-500' : 'text-slate-400'}`} />
                      <span className="text-sm font-semibold text-slate-700 text-center">{cvFile ? cvFile.name : 'Upload CV (.pdf)'}</span>
                      <input id="cv-upload" type="file" accept=".pdf" className="hidden" onChange={(e) => setCvFile(e.target.files[0])} />
                    </div>

                    {/* Job Description — paste textarea (primary) with optional file upload fallback */}
                    <div className="bg-white border border-slate-200 rounded-2xl p-5">
                      <div className="flex items-center justify-between mb-2">
                        <label className="text-sm font-semibold text-slate-700 flex items-center gap-2">
                          <FileText className={`w-4 h-4 ${(jdText.trim() || jdFile) ? 'text-emerald-500' : 'text-slate-400'}`} />
                          Job Description
                        </label>
                        <div className="flex items-center gap-3">
                          <label className="text-xs text-slate-500 hover:text-orange-600 cursor-pointer underline" onClick={() => document.getElementById('jd-upload').click()}>
                            {jdFile ? `Using file: ${jdFile.name}` : 'or upload a .txt file'}
                          </label>
                          {jdFile && (
                            <button type="button" onClick={() => setJdFile(null)} className="text-xs text-slate-400 hover:text-red-500">clear</button>
                          )}
                          <input id="jd-upload" type="file" accept=".txt" className="hidden" onChange={(e) => setJdFile(e.target.files[0])} />
                        </div>
                      </div>
                      <textarea
                        value={jdText}
                        onChange={(e) => setJdText(e.target.value)}
                        disabled={!!jdFile}
                        placeholder={jdFile ? "File selected — clear it to paste text instead." : "Paste the full job description here. Include responsibilities, requirements, and any skills listed..."}
                        rows={6}
                        className={`w-full bg-slate-50 border border-slate-200 rounded-xl p-3 text-sm text-slate-700 placeholder-slate-400 outline-none focus:border-orange-400 focus:bg-white resize-y ${jdFile ? 'opacity-50 cursor-not-allowed' : ''}`}
                      />
                      {jdText.trim() && !jdFile && (
                        <p className="text-xs text-slate-500 mt-2">{jdText.trim().split(/\s+/).length} words pasted</p>
                      )}
                    </div>
                  </div>

                  <div className="flex gap-6 w-full max-w-2xl">
                    <div className="flex-1 bg-white border border-slate-200 rounded-2xl p-4 flex flex-col items-center">
                      <label className="text-sm font-semibold text-slate-600 mb-2">Technical Difficulty</label>
                      <select value={difficulty} onChange={e => setDifficulty(e.target.value)} className="bg-white text-slate-800 p-2 rounded w-full border border-slate-200 outline-none focus:border-orange-400">
                        <option value="Easy">Easy</option>
                        <option value="Medium">Medium</option>
                        <option value="Hard">Hard</option>
                        <option value="Brutal">Brutal</option>
                      </select>
                    </div>
                    <div className="flex-1 bg-white border border-slate-200 rounded-2xl p-4 flex flex-col items-center">
                      <label className="text-sm font-semibold text-slate-600 mb-2">Number of Questions</label>
                      <input type="number" min="1" max="15" value={numQuestions} onChange={e => setNumQuestions(e.target.value)} className="bg-white text-slate-800 p-2 rounded w-full border border-slate-200 text-center outline-none focus:border-orange-400" />
                    </div>
                  </div>

                  <button 
                    onClick={handleUpload}
                    disabled={!cvFile || (!jdFile && !jdText.trim()) || isUploading}
                    className={`mt-8 px-10 py-4 rounded-full font-bold flex items-center gap-3 transition-all ${(!cvFile || (!jdFile && !jdText.trim())) ? 'bg-slate-200 text-slate-400 cursor-not-allowed' : 'bg-gradient-to-r from-orange-500 to-rose-500 hover:from-orange-600 hover:to-rose-600 text-white shadow-lg shadow-orange-500/30'}`}>
                    {isUploading ? <Activity className="w-5 h-5 animate-spin" /> : <CheckCircle className="w-5 h-5" />}
                    {isUploading ? 'Building Vector Database...' : 'Initialize Context Engine'}
                  </button>

                  {uploadError && (
                    <div className="mt-6 max-w-2xl w-full bg-red-50 border border-red-200 rounded-2xl p-4 text-sm text-red-700">
                      <p className="font-semibold mb-1">Backend unreachable</p>
                      <p className="text-red-600">{uploadError}</p>
                    </div>
                  )}
                </div>
              ) : (
                <div className="min-h-[350px] relative overflow-hidden rounded-[2.5rem] bg-white/80 backdrop-blur-xl border border-white shadow-sm p-10 flex flex-col justify-center">
                  {latestFeedback && status !== 'processing' && (
                    <div className="absolute top-0 left-0 w-full bg-gradient-to-r from-orange-50 to-rose-50 border-b border-orange-200 p-3 px-8">
                      <p className="text-orange-800 text-sm font-medium tracking-wide">{latestFeedback}</p>
                    </div>
                  )}
                  {status === 'processing' ? (
                    <div className="flex flex-col items-center justify-center opacity-60">
                      <Activity className="w-12 h-12 text-orange-500 animate-pulse mb-4" />
                      <p className="tracking-widest uppercase text-sm font-bold text-orange-600">Processing Sequence</p>
                    </div>
                  ) : (
                    <p className="text-3xl md:text-4xl leading-[1.4] font-medium text-slate-800 mt-8">
                      "{currentQuestion}"
                    </p>
                  )}
                </div>
              )}

              {/* Control panel */}
              <div className="rounded-[2rem] bg-white/80 backdrop-blur-xl border border-white shadow-sm p-6 flex flex-col items-center justify-center relative overflow-hidden min-h-[140px]">
                {status === 'listening' && (
                   <canvas ref={canvasRef} width="600" height="100" className="absolute inset-0 w-full h-full z-0 opacity-40"></canvas>
                )}
                <div className="relative z-10 w-full flex justify-between items-center px-4">
                  {status === 'setup' && <p className="text-slate-400 text-sm italic w-full text-center">System awaiting data input...</p>}
                  
                  {status === 'idle' && (
                    <button onClick={startInterview} className="mx-auto flex items-center gap-3 bg-gradient-to-r from-orange-500 to-rose-500 text-white px-8 py-4 rounded-full font-bold hover:scale-105 transition-transform shadow-lg shadow-orange-500/30">
                      <PlayCircle className="w-6 h-6" /> Begin {difficulty} Interview
                    </button>
                  )}
                  {status === 'asking' && !isInterviewComplete && (
                    <button onClick={startRecording} className="mx-auto flex items-center gap-3 bg-emerald-50 border border-emerald-300 text-emerald-700 px-8 py-4 rounded-full font-bold hover:bg-emerald-100 transition-colors">
                      <Mic className="w-6 h-6" /> Activate Microphone
                    </button>
                  )}
                  {status === 'listening' && (
                    <button onClick={stopRecording} className="mx-auto flex items-center gap-3 bg-red-50 border border-red-300 text-red-600 px-8 py-4 rounded-full font-bold hover:bg-red-100 transition-colors">
                      <Square className="w-5 h-5 fill-current" /> Terminate Feed & Submit
                    </button>
                  )}
                  {(status === 'asking' || status === 'listening') && transcript.length > 0 && (
                    <button onClick={endInterview} className={`absolute right-4 text-sm font-semibold tracking-wider transition-colors ${isInterviewComplete ? 'text-emerald-600 animate-pulse border border-emerald-300 px-4 py-2 rounded-full bg-emerald-50' : 'text-slate-400 hover:text-slate-700'}`}>
                      {isInterviewComplete ? 'Generate Report' : 'End Session'}
                    </button>
                  )}
                </div>
              </div>
            </div>

            {/* Right column: video + log */}
            <div className="col-span-1 lg:col-span-4 flex flex-col gap-6 h-[600px]">
              
              {status !== 'setup' && status !== 'debrief' && (
                <div className="bg-white/80 backdrop-blur-xl rounded-[2.5rem] border border-white shadow-sm overflow-hidden flex flex-col h-[280px] relative group">
                  <div className="absolute top-4 left-4 z-10 flex items-center gap-2 bg-white/90 backdrop-blur-md px-3 py-1.5 rounded-full border border-slate-200 shadow-sm">
                    <div className={`w-2 h-2 rounded-full ${status === 'listening' ? 'bg-red-500 animate-pulse' : 'bg-emerald-500'}`}></div>
                    <span className="text-xs font-bold text-slate-700 uppercase tracking-wider">
                      {status === 'listening' ? 'Tracking Active' : 'Camera Ready'}
                    </span>
                  </div>
                  {status === 'processing' && (
                    <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-white/80 backdrop-blur-sm">
                      <Activity className="w-8 h-8 text-orange-500 animate-spin mb-3" />
                      <p className="text-xs font-bold uppercase tracking-widest text-orange-600">Analysing Response</p>
                    </div>
                  )}
                  
                  <img 
                    src={`${API_BASE}/video_feed`}
                    alt="AI Behavioral Tracking Feed" 
                    className="w-full h-full object-cover"
                    onError={(e) => {
                      e.target.style.display = 'none';
                      if (e.target.nextSibling) e.target.nextSibling.style.display = 'flex';
                    }}
                  />
                  
                  <div className="hidden absolute inset-0 flex-col items-center justify-center bg-slate-50 text-slate-500 text-sm text-center p-6">
                    <Activity className="w-8 h-8 mb-2 opacity-50" />
                    <p>Behavioral Tracking Offline.<br/>Start the backend video_feed endpoint.</p>
                  </div>
                </div>
              )}

              <div className="flex-1 flex flex-col bg-white/80 backdrop-blur-xl rounded-[2.5rem] border border-white shadow-sm overflow-hidden">
                <div className="p-6 border-b border-slate-100">
                  <h3 className="text-slate-800 font-bold flex items-center gap-3 text-lg">Session Log</h3>
                </div>
                <div className="flex-1 overflow-y-auto p-6 space-y-6 custom-scrollbar">
                  {transcript.length === 0 && <p className="text-slate-400 text-sm italic text-center mt-10">Transcript will populate during the session.</p>}
                  {transcript.map((item, index) => (
                    <div key={index} className="space-y-2 border-b border-slate-100 pb-4 last:border-0">
                      <p className="text-xs text-orange-600 font-bold uppercase">System</p>
                      <p className="text-sm text-slate-700 line-clamp-2">{item.q}</p>
                      <p className="text-xs text-emerald-600 font-bold uppercase mt-3">Candidate</p>
                      <p className="text-sm text-slate-500 italic">"{item.a}"</p>
                    </div>
                  ))}
                </div>
              </div>

            </div>
          </main>
        </div>
      </div>
    </div>
  );
}