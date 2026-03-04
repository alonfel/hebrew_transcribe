"""
Stub out mlx_whisper for platforms where it can't be installed (e.g. Linux CI).
This runs before any test module imports, so `import mlx_whisper` in transcribe.py
resolves to this stub instead of failing.
"""
import sys
import types

if "mlx_whisper" not in sys.modules:
    stub = types.ModuleType("mlx_whisper")
    stub.transcribe = lambda *args, **kwargs: {"segments": []}
    sys.modules["mlx_whisper"] = stub
