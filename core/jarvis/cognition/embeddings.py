"""Embeddings de frases para la segunda etapa del clasificador.

El modelo es paraphrase-multilingual-MiniLM-L12-v2 en ONNX cuantizado para
AVX2: 118 MB, ~6 ms por frase en el i5-10500. Corre con onnxruntime y
tokenizers, que ya estan instalados por el pipeline de voz: no hace falta
PyTorch.

DECISION: multilingue y no el all-MiniLM ingles que sugiere el documento,
porque las ordenes llegan en espanol. El ingles da similitudes casi aleatorias
entre "pon algo de musica" y "reproduce una cancion".
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

from jarvis.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

__all__ = [
    "MODEL_FILE",
    "TOKENIZER_FILE",
    "Embedder",
    "EmbeddingModelMissingError",
    "OnnxEmbedder",
    "cached_embeddings",
]

_log = get_logger(__name__)

MODEL_FILE = "model.onnx"
TOKENIZER_FILE = "tokenizer.json"
_MAX_TOKENS = 32
"""La intencion de una orden se decide en sus primeras palabras. Medido: con 64
una frase larga tardaba 15 ms; con 32 el peor caso queda dentro de los 10 ms."""
_THREADS = 2
"""Medido: con 2 hilos la latencia es la misma que con 6 y deja CPU a Whisper."""


class EmbeddingModelMissingError(RuntimeError):
    """Falta el modelo de embeddings. El mensaje dice como conseguirlo."""

    def __init__(self, path: Path) -> None:
        """Construye el error con la ruta esperada."""
        super().__init__(
            f"Falta el modelo de intenciones en {path}. Descargalo con: "
            ".venv/Scripts/python scripts/download_cognition_models.py"
        )


class Embedder(Protocol):
    """Convierte frases en vectores normalizados (norma 1)."""

    @property
    def identity(self) -> str:
        """Identifica al modelo; cambia si cambian los pesos. Llave de la cache."""
        ...

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Una fila por frase, ya normalizada: el producto punto es el coseno."""
        ...


class OnnxEmbedder:
    """MiniLM en onnxruntime, con media de tokens como en sentence-transformers."""

    def __init__(self, model_dir: Path, max_tokens: int = _MAX_TOKENS) -> None:
        """Carga el modelo y el tokenizador. Lanza si falta alguno.

        Args:
            model_dir: carpeta con `model.onnx` y `tokenizer.json`.
            max_tokens: donde se trunca cada frase. 32 para ordenes; la memoria
                usa mas, porque un hecho dice lo importante al final.
        """
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_path = model_dir / MODEL_FILE
        tokenizer_path = model_dir / TOKENIZER_FILE
        for path in (model_path, tokenizer_path):
            if not path.exists():
                raise EmbeddingModelMissingError(path)

        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.enable_truncation(max_tokens)
        self._tokenizer.enable_padding()
        options = ort.SessionOptions()
        options.intra_op_num_threads = _THREADS
        self._session = ort.InferenceSession(
            str(model_path), options, providers=["CPUExecutionProvider"]
        )
        self._needs_token_types = any(
            node.name == "token_type_ids" for node in self._session.get_inputs()
        )
        stat = model_path.stat()
        self._identity = f"{model_path.name}:{stat.st_size}:{max_tokens}"

    def with_max_tokens(self, max_tokens: int) -> OnnxEmbedder:
        """Otra vista del mismo modelo con otro truncado, sin cargarlo dos veces.

        DECISION: la memoria y el clasificador comparten la sesion de ONNX. Son
        118 MB de pesos; cargarlos dos veces no compra nada.
        """
        from tokenizers import Tokenizer

        twin = copy.copy(self)
        twin._tokenizer = Tokenizer.from_str(self._tokenizer.to_str())
        twin._tokenizer.enable_truncation(max_tokens)
        twin._identity = self._identity.rsplit(":", 1)[0] + f":{max_tokens}"
        return twin

    @property
    def identity(self) -> str:
        """Nombre y tamano del archivo: basta para invalidar la cache."""
        return self._identity

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Codifica un lote de frases."""
        encodings = self._tokenizer.encode_batch(list(texts))
        ids = np.array([e.ids for e in encodings], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if self._needs_token_types:
            feeds["token_type_ids"] = np.zeros_like(ids)
        hidden = np.asarray(self._session.run(None, feeds)[0], dtype=np.float32)
        weights = mask[..., None].astype(np.float32)
        pooled = (hidden * weights).sum(axis=1) / np.maximum(weights.sum(axis=1), 1.0)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return (pooled / np.maximum(norms, 1e-9)).astype(np.float32)


def cached_embeddings(
    embedder: Embedder,
    texts: Sequence[str],
    cache_dir: Path | None,
) -> NDArray[np.float32]:
    """Embeddings de los ejemplos del catalogo, con cache en disco.

    Codificar los ~300 ejemplos cuesta ~2 s al arrancar. La cache se invalida
    sola si cambia el catalogo o el modelo, porque ambos entran en el hash.
    """
    digest = hashlib.sha256()
    digest.update(embedder.identity.encode())
    for text in texts:
        digest.update(b"\x00" + text.encode())
    path = cache_dir / f"catalog-{digest.hexdigest()[:16]}.npy" if cache_dir else None
    if path is not None and path.exists():
        try:
            cached: NDArray[np.float32] = np.asarray(np.load(path), dtype=np.float32)
            if cached.shape[0] == len(texts):
                return cached
        except (OSError, ValueError) as error:
            _log.warning("intent.cache_unreadable", path=str(path), error=str(error))
    vectors = embedder.embed(texts)
    if path is not None:
        try:
            for stale in path.parent.glob("catalog-*.npy"):
                stale.unlink()
            np.save(path, vectors)
        except OSError as error:
            _log.warning("intent.cache_unwritable", path=str(path), error=str(error))
    return vectors
