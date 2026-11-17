"""LLM backend plugins. Any OpenAI-compatible endpoint works — including
fully local servers (Ollama, llama.cpp server, vLLM, LM Studio) — keeping the
agent local-first. Drop-in alternative providers only need to satisfy the
``LLMProvider`` Protocol and expose the standard plugin discovery contract.
"""
