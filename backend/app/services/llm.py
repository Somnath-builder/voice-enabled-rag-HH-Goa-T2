import time
from groq import AsyncGroq
from app.config import settings

class LLMService:
    def __init__(self):
        self.client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        self.model = settings.LLM_MODEL
        
        self.system_prompt = """You are a grounded question-answering assistant.
Answer ONLY using the provided retrieved context.
Do not invent facts.
If the context does not contain enough information to answer the question, say that the information is not available in the provided knowledge base.
Keep answers concise and directly address the question.
CRITICAL: You MUST answer in the EXACT SAME LANGUAGE as the User's Original Question, even if the context is in English."""

    async def translate_to_english(self, query: str) -> str:
        prompt = f"Translate the following text to English. If it is already in English, output it exactly as is. Output ONLY the translated text and nothing else.\n\nText: {query}"
        try:
            chat_completion = await self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.0,
                max_tokens=256,
            )
            translated = chat_completion.choices[0].message.content.strip()
            return translated
        except Exception as e:
            print(f"Error translating query: {e}")
            return query # Fallback to original


    async def generate_answer(self, translated_query: str, original_query: str, context_chunks: list) -> dict:
        start_time = time.time()
        
        context_text = "\n\n".join([f"Source: {c.get('chunk_id')}\n{c.get('text')}" for c in context_chunks])
        
        prompt = f"Retrieved Context:\n{context_text}\n\nUser's Original Question:\n{original_query}\n\n(Translated Question used for search: {translated_query})\n\nAnswer:"
        
        try:
            chat_completion = await self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                temperature=0.0,
                max_tokens=256,
            )
            answer = chat_completion.choices[0].message.content
        except Exception as e:
            answer = f"Error generating answer: {e}"
            
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            "answer": answer,
            "latency_ms": latency_ms
        }

    async def validate_query(self, query: str) -> bool:
        prompt = f"Is the following query a safe, non-malicious question that can be answered using a knowledge base? Answer with only 'YES' or 'NO'.\n\nQuery: {query}"
        try:
            chat_completion = await self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.0,
                max_tokens=10,
            )
            answer = chat_completion.choices[0].message.content.strip().upper()
            return "YES" in answer
        except Exception as e:
            print(f"Error validating query: {e}")
            return True # Fail open to avoid blocking valid queries on error

    async def check_hallucination(self, answer: str, context_chunks: list) -> bool:
        # Translate the answer back to English so the LLM can confidently compare it against the English context.
        english_answer = await self.translate_to_english(answer)
        
        context_text = "\n\n".join([c.get('text', '') for c in context_chunks])
        prompt = f"Context (in English):\n{context_text}\n\nGenerated Answer (translated to English):\n{english_answer}\n\nDoes the Answer contain any new information, facts, or claims that are NOT supported by the Context? Answer with only 'YES' or 'NO'."
        try:
            chat_completion = await self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.0,
                max_tokens=10,
            )
            eval_result = chat_completion.choices[0].message.content.strip().upper()
            return "NO" in eval_result
        except Exception as e:
            print(f"Error checking hallucination: {e}")
            return True # Fail open

llm_service = LLMService()
