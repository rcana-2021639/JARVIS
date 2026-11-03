"""Utilidades de streaming: JSON incremental y corte en frases hablables.

`StructuredStream` lee el JSON del nivel 2 mientras se genera. El esquema pone
`confianza` antes que `respuesta`: en cuanto llega el numero se decide si
hablar, esperar o escalar, y el texto de la respuesta se extrae caracter a
caracter sin esperar a que cierre el objeto.

`SentenceChunker` convierte el texto que llega en trozos que se pueden decir:
frases completas, nunca codigo, y no mas de las que la personalidad permite
decir en voz alta.
"""

from __future__ import annotations

import json
import re
from typing import Literal

__all__ = ["SentenceChunker", "StructuredStream"]

Phase = Literal["prefix", "confidence", "between", "answer", "done"]

_CONFIDENCE = re.compile(r'"confianza"\s*:\s*(-?\d+(?:\.\d+)?)\s*[,}\s]')
_CONFIDENCE_KEY = re.compile(r'"confianza"\s*:')
_ANSWER_START = re.compile(r'"respuesta"\s*:\s*"')
_ESCAPES = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}


class StructuredStream:
    """Parser incremental de `{"confianza": x, "respuesta": "..."}`."""

    def __init__(self) -> None:
        """Arranca vacio."""
        self._buffer = ""
        self._answer_start: int | None = None
        self._cursor = 0
        self.confidence: float | None = None
        self.answer = ""
        self.answer_closed = False

    @property
    def raw(self) -> str:
        """Todo lo recibido."""
        return self._buffer

    @property
    def phase(self) -> Phase:
        """En que parte del objeto va la generacion."""
        if self.answer_closed:
            return "done"
        if self._answer_start is not None:
            return "answer"
        if self.confidence is not None:
            return "between"
        if _CONFIDENCE_KEY.search(self._buffer):
            return "confidence"
        return "prefix"

    def feed(self, delta: str) -> str:
        """Agrega texto generado. Devuelve lo nuevo de la respuesta, ya decodificado."""
        self._buffer += delta
        if self.confidence is None:
            match = _CONFIDENCE.search(self._buffer)
            if match:
                self.confidence = min(1.0, max(0.0, float(match.group(1))))
        if self._answer_start is None:
            match = _ANSWER_START.search(self._buffer)
            if match is None:
                return ""
            self._answer_start = self._cursor = match.end()
        return self._decode()

    def _decode(self) -> str:
        out: list[str] = []
        text = self._buffer
        i = self._cursor
        while i < len(text) and not self.answer_closed:
            char = text[i]
            if char == '"':
                self.answer_closed = True
                i += 1
                break
            if char != "\\":
                out.append(char)
                i += 1
                continue
            if i + 1 >= len(text):
                break  # escape a medias: esperar al siguiente trozo
            code = text[i + 1]
            if code == "u":
                if i + 6 > len(text):
                    break
                out.append(chr(int(text[i + 2 : i + 6], 16)))
                i += 6
            else:
                out.append(_ESCAPES.get(code, code))
                i += 2
        self._cursor = i
        piece = "".join(out)
        self.answer += piece
        return piece

    def finish(self) -> tuple[float | None, str]:
        """Al terminar: intenta un parseo completo por si el incremental fallo."""
        if not self.answer_closed:
            try:
                data = json.loads(self._buffer)
            except ValueError:
                return self.confidence, self.answer
            if isinstance(data, dict):
                answer = data.get("respuesta")
                confidence = data.get("confianza")
                if isinstance(answer, str):
                    self.answer = answer
                if isinstance(confidence, int | float):
                    self.confidence = min(1.0, max(0.0, float(confidence)))
        return self.confidence, self.answer


_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+|\n+")
_CODE_FENCE = "```"
_COMPLETE = re.compile(r"[.!?…]\s*$")


class SentenceChunker:
    """Corta texto en streaming en frases completas para el TTS.

    Deja de entregar cuando aparece un bloque de codigo o cuando ya se dijeron
    `max_sentences`: lo que sigue va al chat, no a los oidos. `overflowed`
    avisa que hubo mas de lo que se dijo.
    """

    def __init__(self, max_sentences: int) -> None:
        """Args: max_sentences, tope de frases habladas."""
        self._max = max_sentences
        self._pending = ""
        self._spoken = 0
        self._stopped = False
        self.overflowed = False

    def feed(self, delta: str) -> list[str]:
        """Agrega texto; devuelve las frases que ya estan completas."""
        if self._stopped:
            if delta.strip():
                self.overflowed = True
            return []
        self._pending += delta
        fence = self._pending.find(_CODE_FENCE)
        if fence >= 0:
            before = self._pending[:fence]
            out = self._emit(before, final=True)
            # Lo que sigue es codigo: existe, pero va al chat.
            self.overflowed = True
            self._stop()
            return out
        return self._emit(self._pending, final=False)

    def salvage(self) -> list[str]:
        """La generacion se corto: lo pendiente se dice solo si es una frase completa."""
        if self._stopped or not _COMPLETE.search(self._pending):
            return []
        return self._emit(self._pending, final=True)

    def flush(self) -> list[str]:
        """Al terminar la generacion: lo que quedo pendiente."""
        if self._stopped:
            return []
        return self._emit(self._pending, final=True)

    def _emit(self, text: str, *, final: bool) -> list[str]:
        parts = _SENTENCE_END.split(text)
        complete = parts if final else parts[:-1]
        self._pending = "" if final else parts[-1]
        out: list[str] = []
        for part in complete:
            sentence = part.strip()
            if not sentence:
                continue
            if self._spoken >= self._max:
                self.overflowed = True
                self._stop()
                break
            out.append(sentence)
            self._spoken += 1
        return out

    def _stop(self) -> None:
        if self._pending.strip():
            self.overflowed = True
        self._stopped = True
        self._pending = ""
