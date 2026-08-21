import time
import json
import asyncio
from groq import AsyncGroq
from app.config import settings

class LLMService:
    def __init__(self):
        self.client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        self.model = settings.LLM_MODEL
        
        self.system_prompt = """You are a highly efficient, grounded question-answering assistant.
You MUST output your response in strict JSON format.

Your JSON output must match exactly this schema:
{
  "is_safe": boolean, // False if the user query is unsafe, malicious, or highly inappropriate. Otherwise True.
  "is_relevant": boolean, // False if the provided Context DOES NOT contain enough information to answer the user query. True if it does.
  "answer": string, // Your answer to the query based strictly on the context. If is_safe or is_relevant is false, leave this blank.
  "confidence_score": float // A score between 0.0 and 1.0 indicating how confident you are that the answer is perfectly grounded in the context.
}

CRITICAL RULES:
1. Answer ONLY using the provided retrieved context. Do not invent facts.
2. You MUST answer in the EXACT SAME LANGUAGE as the User's Original Question, even if the context is in English.
3. Keep answers concise and directly address the question.
4. Output ONLY valid JSON. No markdown formatting, no code blocks."""

    async def orchestrated_generation(self, query: str, context_chunks: list) -> dict:
        start_time = time.time()
        
        context_text = "\n\n".join([f"Source: {c.get('chunk_id')}\n{c.get('text')}" for c in context_chunks])
        
        prompt = f"Retrieved Context:\n{context_text}\n\nUser's Original Question:\n{query}"
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                chat_completion = await self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    model=self.model,
                    temperature=0.0,
                    max_tokens=512,
                    response_format={"type": "json_object"}
                )
                
                output_str = chat_completion.choices[0].message.content
                parsed_output = json.loads(output_str)
                
                # Enforce required keys
                required_keys = ["is_safe", "is_relevant", "answer", "confidence_score"]
                for key in required_keys:
                    if key not in parsed_output:
                        parsed_output[key] = None
                        
                latency_ms = (time.time() - start_time) * 1000
                parsed_output["latency_ms"] = latency_ms
                
                return parsed_output
                
            except json.JSONDecodeError as e:
                print(f"JSON Parsing Error on attempt {attempt+1}: {e}")
                if attempt == max_retries - 1:
                    return {"is_safe": True, "is_relevant": False, "answer": "", "confidence_score": 0.0, "latency_ms": (time.time() - start_time) * 1000, "error": "JSON parsing failed after retries."}
                await asyncio.sleep(0.5 * (attempt + 1))
            except Exception as e:
                print(f"Error generating answer on attempt {attempt+1}: {e}")
                if attempt == max_retries - 1:
                    return {"is_safe": True, "is_relevant": False, "answer": "", "confidence_score": 0.0, "latency_ms": (time.time() - start_time) * 1000, "error": str(e)}
                await asyncio.sleep(0.5 * (attempt + 1))

llm_service = LLMService()

