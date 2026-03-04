"""
Remove the mlx_whisper stub injected by tests/conftest.py so integration
tests use the real mlx_whisper installed in the environment.
"""
import sys

sys.modules.pop("mlx_whisper", None)
