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
Keep answers concise and directly address the question."""

    async def generate_answer(self, query: str, context_chunks: list) -> dict:
        start_time = time.time()
        
        context_text = "\n\n".join([f"Source: {c.get('chunk_id')}\n{c.get('text')}" for c in context_chunks])
        
        prompt = f"Retrieved Context:\n{context_text}\n\nUser Question:\n{query}"
        
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

llm_service = LLMService()
