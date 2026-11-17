"""Microkernel primitives: event bus, plugin lifecycle, interfaces, tool registry.

Everything in echo-agent hangs off these four modules. The orchestrator and all
plugins depend *only* on this layer — never on each other — which is what keeps
the system infinitely expandable: a new capability is a new plugin, a new event
topic, and (optionally) a new tool schema.
"""
