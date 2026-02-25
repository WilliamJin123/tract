"""Dummy module used by 08_tool_calling.py as a search target."""

import math

# DISCOVERY: Tool calls are just commits with metadata — the context
# window is the source of truth.

def compute_area(radius: float) -> float:
    return math.pi * radius ** 2

def greet(name: str) -> str:
    return f"Hello, {name}!"
