"""
chatbot.py
Saare pieces ko jodta hai — yahi main brain hai chatbot ka.

Flow:
User query → Retriever (relevant chunks) → Groq LLM → Answer
"""

from retriever import GlaRetriever
from groq_llm import GroqLLM


class GLAChatbot:
    """
    GLA Admission Chatbot — retriever + LLM ka combination.

    Usage:
        bot = GLAChatbot()
        answer = bot.ask("What are the B.Tech fees?")
        print(answer)
    """

    def __init__(
        self,
        vector_store_dir: str = "vector_store",
        model: str = "llama-3.1-8b-instant"
    ):
        print("GLA Chatbot initialize ho raha hai...")
        self.retriever = GlaRetriever(vector_store_dir)
        self.llm = GroqLLM(model=model)
        self.chat_history: list[dict] = []
        print("  [ok] Chatbot ready!\n")

    def ask(self, user_query: str, use_history: bool = True) -> str:
        """
        User ka sawal lo, answer wapas do.
        """
        if not user_query.strip():
            return "Please ask a question!"

        context = self.retriever.retrieve_and_format(user_query)

        if use_history and self.chat_history:
            answer = self.llm.generate_with_history(
                user_query, context, self.chat_history
            )
        else:
            answer = self.llm.generate(user_query, context)

        self.chat_history.append({"role": "user", "content": user_query})
        self.chat_history.append({"role": "assistant", "content": answer})
        if len(self.chat_history) > 20:
            self.chat_history = self.chat_history[-20:]

        return answer

    def ask_with_sources(self, user_query: str) -> dict:
        """
        Answer ke saath source chunks bhi chahiye toh ye use karo.

        Returns:
            {
                "answer": "...",
                "sources": ["brochure.pdf", ...],
                "chunks": ["chunk text...", ...]
            }
        """
        docs = self.retriever.get_relevant_chunks(user_query)
        context = self.retriever.format_context(docs)
        answer = self.llm.generate(user_query, context)

        return {
            "answer": answer,
            "sources": list({doc.metadata.get("source", "?") for doc in docs}),
            "chunks": [doc.page_content[:200] for doc in docs],
        }

    def reset_history(self):
        """Conversation history clear karo."""
        self.chat_history = []
        print("Chat history cleared.")

    def get_history(self) -> list[dict]:
        """Current conversation history return karo."""
        return self.chat_history.copy()


def run_cli():
    """Terminal mein chatbot chalao."""
    print("=" * 55)
    print("  GLA University Admission Chatbot")
    print("  Type 'quit' to exit | 'clear' to reset history")
    print("=" * 55)

    bot = GLAChatbot()

    while True:
        try:
            query = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nChatbot closed. Goodbye!")
            break

        if not query:
            continue

        if query.lower() in ("quit", "exit", "bye"):
            print("Thank you! Welcome to GLA University. 🎓")
            break

        if query.lower() == "clear":
            bot.reset_history()
            continue

        answer = bot.ask(query)
        print(f"\nChatbot: {answer}")


if __name__ == "__main__":
    run_cli()
