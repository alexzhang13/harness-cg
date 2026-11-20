"""Re-export of the guarded shared router (synth-datagen/ctcbench/shared_router.py)."""
import sys
sys.path.insert(0, __import__("os").path.abspath(__file__ + "/../.."))
from shared_router import ROUTER, RoutedAsyncOpenAI, Router  # noqa: E402,F401
