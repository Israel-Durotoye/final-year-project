"""Version-tolerant Keras helpers shared by every LSTM inference module.

The deployed environment runs Keras 2.x on TensorFlow, while some artifacts
were produced with Keras 3. ``keras.saving.load_model`` exists only in Keras 3,
so the inference modules must not call it directly.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

# Honoured by Keras 3 only; Keras 2 ignores it and uses TensorFlow.
os.environ.setdefault("KERAS_BACKEND", "torch")

# Keras models are not safe to call from several request threads at once.
PREDICT_LOCK = threading.Lock()


def load_keras_model(path: str | os.PathLike[str]) -> Any:
    """Load a saved model for inference under Keras 2 or Keras 3."""
    import keras  # noqa: PLC0415 - heavy import, intentionally lazy

    return keras.models.load_model(str(Path(path)), compile=False)


def predict(model: Any, batch: Any) -> Any:
    """Run one thread-safe forward pass."""
    with PREDICT_LOCK:
        return model.predict(batch, verbose=0)
