"""echo-agent · local-first, self-improving, event-driven AI agent.

Package layout:
    python_core.core          microkernel: event bus, plugins, interfaces, tools
    python_core.orchestrator  the main agent loop (token pruning, LLM routing)
    python_core.plugins       feature plugins (memory, self-healing, skills, gateways)
    python_core.inter_agent   universal HTTP/JSON-RPC agent protocol layer
"""

__version__ = "0.2.0"
