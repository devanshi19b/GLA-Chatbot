"""
groq_llm.py
Groq API se fast LLM response generate karta hai.
Model: llama-3.1-8b-instant (fast)
Fallback: llama-3.3-70b-versatile (accurate)
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()


DEFAULT_MODEL = "llama-3.1-8b-instant"
FALLBACK_MODEL = "llama-3.3-70b-versatile"
MAX_TOKENS = 1024
TEMPERATURE = 0.0   # 0 = fully deterministic, numbers never change


SYSTEM_PROMPT = """You are the official admission assistant for GLA University.

Rules:
1. Answer ONLY from the provided context — do not add anything from your own knowledge.
2. Copy numbers, fees, and percentages EXACTLY as written in the context — never modify them.
3. If something is clearly stated in the context, present it in the same order as in the context.
4. If confused about which fee belongs to which course — quote the exact line from context.
5. If information is not available say: "This information was not found in the brochure."
6. Always respond in English only.
7. Use bullet points for lists.
8. Never swap or mix up values between different courses or programs.
"""


class GroqLLM:
    """
    Groq API wrapper — prompt build karo aur response lo.
    """

    def __init__(self, model: str = DEFAULT_MODEL):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY nahi mili!\n"
                ".env file mein daalo: GROQ_API_KEY=gsk_..."
            )

        self.api_key = api_key
        self.model = model
        print(f"  [ok] Groq LLM ready ({self.model})")

    def build_prompt(self, user_query: str, context: str) -> list[dict]:
        """
        Messages array banao Groq API ke format mein.
        """
        user_message = f"""Below is the context extracted from GLA University's official brochures.
Answer my question strictly based on this context only.

=== CONTEXT (from PDF) ===
{context}
==========================

My question: {user_query}"""

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

    def generate(self, user_query: str, context: str) -> str:
        """
        Main function — query + context do, answer lo.
        """
        messages = self.build_prompt(user_query, context)

        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": MAX_TOKENS,
                    "temperature": TEMPERATURE,
                },
                timeout=60,
            )
            response.raise_for_status()
            payload = response.json()
            return payload["choices"][0]["message"]["content"].strip()

        except Exception as e:
            if self.model != FALLBACK_MODEL:
                print(f"  [warn] {self.model} failed, trying {FALLBACK_MODEL}...")
                self.model = FALLBACK_MODEL
                return self.generate(user_query, context)
            else:
                return f"Sorry, unable to generate a response right now. Error: {str(e)}"

    def generate_with_history(
        self,
        user_query: str,
        context: str,
        chat_history: list[dict]
    ) -> str:
        """
        Multi-turn conversation ke liye — history bhi bhejo.
        """
        system_msg = {"role": "system", "content": SYSTEM_PROMPT}

        context_msg = {
            "role": "user",
            "content": f"=== CONTEXT ===\n{context}\n==============="
        }
        context_ack = {
            "role": "assistant",
            "content": "Understood. Please ask your question."
        }

        messages = [system_msg, context_msg, context_ack] + chat_history + [
            {"role": "user", "content": user_query}
        ]

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        return payload["choices"][0]["message"]["content"].strip()


if __name__ == "__main__":
    llm = GroqLLM()

    test_context = """
    [Source 1: gla_btech.pdf]
    B. Tech CSE (Core) | 2.0L | 2.08L | 2.16L | 2.24L
    B. Tech CSE (AIML) | 2.35L | 2.43L | 2.51L | 2.59L
    B. Tech CSE (Honors) | 2.40L | 2.45L | 2.50L | 2.55L
    """

    query = "What are the fees for B.Tech CSE programs?"
    answer = llm.generate(query, test_context)

    print(f"Query: {query}")
    print(f"\nAnswer:\n{answer}")
