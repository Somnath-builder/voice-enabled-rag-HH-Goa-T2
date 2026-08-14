"use client";

import { useState, useRef } from "react";
import { Mic, Square, Loader2 } from "lucide-react";

interface AudioRecorderProps {
  onAudioReady: (file: File) => void;
  isProcessing: boolean;
}

export default function AudioRecorder({ onAudioReady, isProcessing }: AudioRecorderProps) {
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorder = useRef<MediaRecorder | null>(null);
  const audioChunks = useRef<Blob[]>([]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      mediaRecorder.current = recorder;
      audioChunks.current = [];

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunks.current.push(event.data);
        }
      };

      recorder.onstop = () => {
        const audioBlob = new Blob(audioChunks.current, { type: "audio/wav" });
        const file = new File([audioBlob], "recording.wav", { type: "audio/wav" });
        onAudioReady(file);
        
        // Stop all tracks to release the microphone
        stream.getTracks().forEach(track => track.stop());
      };

      recorder.start();
      setIsRecording(true);
    } catch (err) {
      console.error("Error accessing microphone:", err);
      alert("Could not access microphone. Please ensure permissions are granted.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorder.current && isRecording) {
      mediaRecorder.current.stop();
      setIsRecording(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center p-6 border border-zinc-800 rounded-xl bg-zinc-900/50 shadow-2xl">
      <div className="mb-6 text-center">
        <h2 className="text-xl font-semibold text-white mb-2">Voice Assistant</h2>
        <p className="text-sm text-zinc-400">
          {isRecording ? "Listening..." : "Tap the microphone and ask a question"}
        </p>
      </div>

      <button
        onClick={isRecording ? stopRecording : startRecording}
        disabled={isProcessing}
        className={`
          relative flex items-center justify-center w-24 h-24 rounded-full transition-all duration-300
          ${isRecording ? "bg-red-500 hover:bg-red-600 scale-110 shadow-[0_0_30px_rgba(239,68,68,0.5)]" : "bg-blue-600 hover:bg-blue-700"}
          ${isProcessing ? "opacity-50 cursor-not-allowed bg-zinc-700 hover:bg-zinc-700" : ""}
        `}
      >
        {isProcessing ? (
          <Loader2 className="w-10 h-10 text-white animate-spin" />
        ) : isRecording ? (
          <Square className="w-8 h-8 text-white fill-white" />
        ) : (
          <Mic className="w-10 h-10 text-white" />
        )}
        
        {/* Pulsing rings when recording */}
        {isRecording && (
          <>
            <span className="absolute w-full h-full rounded-full border-2 border-red-500 animate-ping opacity-75"></span>
            <span className="absolute w-[120%] h-[120%] rounded-full border border-red-500/50 animate-ping opacity-50" style={{ animationDelay: '0.2s' }}></span>
          </>
        )}
      </button>
    </div>
  );
}
