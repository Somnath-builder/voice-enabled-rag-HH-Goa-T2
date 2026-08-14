import httpx
import time
from typing import Dict, Any, Optional
from fastapi import UploadFile
from app.config import settings

class VoiceService:
    def __init__(self):
        self.stt_url = "https://api.sarvam.ai/speech-to-text"
        
    async def transcribe(self, audio_file: UploadFile, language_code: str = "hi-IN") -> Dict[str, Any]:
        """
        Transcribe an uploaded audio file using Sarvam AI's STT API.
        Accepts language_code like "hi-IN" or "en-IN".
        """
        start_time = time.time()
        
        headers = {
            "api-subscription-key": settings.SARVAM_API_KEY
        }
        
        # Sarvam API expects multipart/form-data
        # with 'file' and 'model' (and optionally 'language_code')
        files = {
            'file': (audio_file.filename, await audio_file.read(), audio_file.content_type)
        }
        data = {
            "model": "saaras:v3"
        }
        # If the API requires language_code to be passed in data (optional for some models, required for others)
        # saaras:v1 is generally indic-aware, but you can pass prompt or language config based on Sarvam's docs.
        # We will pass language_code in case it's required.
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.stt_url,
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=30.0
                )
                
            if response.status_code != 200:
                print(f"STT Error: HTTP {response.status_code} - {response.text}")
            
            response.raise_for_status()
            result = response.json()
            
            transcript = result.get("transcript", "")
            if not transcript:
                print(f"Warning: Empty transcript from STT API. Response: {result}")
                
        except Exception as e:
            print(f"STT Error: {e}")
            transcript = ""
            
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            "transcript": transcript,
            "latency_ms": latency_ms
        }

voice_service = VoiceService()
