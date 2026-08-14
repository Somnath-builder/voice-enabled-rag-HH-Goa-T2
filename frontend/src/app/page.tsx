"use client";

import { useState } from "react";
import AudioRecorder from "@/components/AudioRecorder";
import { AlertCircle, CheckCircle2, Clock, Volume2, Search, Database } from "lucide-react";

interface Source {
  chunk_id: string;
  score: number;
  text: string;
}

interface VoiceResponse {
  transcript: string;
  answer: string;
  grounded: boolean;
  sources: Source[];
  stt_latency_ms: number;
  retrieval_latency_ms: number;
  generation_latency_ms: number;
  total_latency_ms: number;
}

export default function Dashboard() {
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<VoiceResponse | null>(null);
  const [showSources, setShowSources] = useState(false);

  const handleAudioReady = async (file: File) => {
    setIsProcessing(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("language_code", "en-IN");

    try {
      const res = await fetch("/api/voice", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || "Failed to process audio");
      }

      const data = await res.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <main className="min-h-screen bg-black text-white p-8 font-sans selection:bg-blue-500/30">
      <div className="max-w-4xl mx-auto space-y-8">
        
        {/* Header */}
        <header className="text-center space-y-4 pt-12 pb-8">
          <div className="inline-flex items-center justify-center p-3 bg-blue-500/10 rounded-2xl mb-4">
            <Volume2 className="w-8 h-8 text-blue-400" />
          </div>
          <h1 className="text-4xl md:text-5xl font-bold tracking-tight bg-gradient-to-br from-white to-zinc-400 bg-clip-text text-transparent">
            Voice RAG Agent
          </h1>
          <p className="text-zinc-400 text-lg max-w-xl mx-auto">
            Ask a question using your voice. The system will transcribe, search the knowledge base, and generate a grounded answer.
          </p>
        </header>

        {/* Recorder Section */}
        <div className="flex justify-center mb-12">
          <div className="w-full max-w-md">
            <AudioRecorder onAudioReady={handleAudioReady} isProcessing={isProcessing} />
          </div>
        </div>

        {/* Error State */}
        {error && (
          <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-xl flex items-start gap-3 text-red-400">
            <AlertCircle className="w-5 h-5 shrink-0 mt-0.5" />
            <p>{error}</p>
          </div>
        )}

        {/* Results Section */}
        {result && (
          <div className="space-y-6 animate-in slide-in-from-bottom-4 duration-500 fade-in">
            
            {/* Transcript */}
            <div className="p-6 rounded-2xl bg-zinc-900 border border-zinc-800">
              <h3 className="text-sm font-medium text-zinc-400 mb-2 uppercase tracking-wider flex items-center gap-2">
                <Volume2 className="w-4 h-4" /> You Said
              </h3>
              <p className="text-lg text-white font-medium leading-relaxed">
                "{result.transcript}"
              </p>
            </div>

            {/* Answer */}
            <div className="p-6 rounded-2xl bg-blue-950/20 border border-blue-900/30 relative overflow-hidden">
              <div className="absolute top-0 left-0 w-1 h-full bg-blue-500"></div>
              
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-medium text-blue-400 uppercase tracking-wider flex items-center gap-2">
                  <Database className="w-4 h-4" /> Generated Answer
                </h3>
                {result.grounded ? (
                  <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                    <CheckCircle2 className="w-3.5 h-3.5" /> Grounded
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-amber-500/10 text-amber-400 border border-amber-500/20">
                    <AlertCircle className="w-3.5 h-3.5" /> Hallucination Risk
                  </span>
                )}
              </div>
              
              <p className="text-lg text-zinc-200 leading-relaxed">
                {result.answer}
              </p>
            </div>

            {/* Metrics & Sources Toggle */}
            <div className="flex flex-wrap items-center gap-4 text-sm text-zinc-500 pt-2">
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-zinc-900 border border-zinc-800">
                <Clock className="w-4 h-4" />
                <span>STT: {(result.stt_latency_ms / 1000).toFixed(2)}s</span>
              </div>
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-zinc-900 border border-zinc-800">
                <Search className="w-4 h-4" />
                <span>RAG: {((result.retrieval_latency_ms + result.generation_latency_ms) / 1000).toFixed(2)}s</span>
              </div>
              <div className="flex-1"></div>
              <button 
                onClick={() => setShowSources(!showSources)}
                className="text-blue-400 hover:text-blue-300 transition-colors text-sm font-medium"
              >
                {showSources ? "Hide Sources" : `View Sources (${result.sources.length})`}
              </button>
            </div>

            {/* Sources */}
            {showSources && result.sources.length > 0 && (
              <div className="mt-4 space-y-3 animate-in fade-in duration-300">
                {result.sources.map((source, idx) => (
                  <div key={idx} className="p-4 rounded-xl bg-zinc-900/50 border border-zinc-800 text-sm">
                    <div className="flex justify-between items-center mb-2 text-zinc-400">
                      <span className="font-mono text-xs">{source.chunk_id}</span>
                      <span className="text-xs bg-zinc-800 px-2 py-0.5 rounded">Score: {source.score.toFixed(3)}</span>
                    </div>
                    <p className="text-zinc-300 leading-relaxed line-clamp-3 hover:line-clamp-none transition-all">
                      {source.text}
                    </p>
                  </div>
                ))}
              </div>
            )}
            
          </div>
        )}

      </div>
    </main>
  );
}
