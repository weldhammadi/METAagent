"""Auto-generated agent by Orchestrator."""
from __future__ import annotations

import os
import json
import uuid
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from streamlit_webrtc import AudioProcessorBase, WebRtcMode, webrtc_streamer

# Load environment variables
load_dotenv()


def get_user_text() -> Dict[str, str]:
    """Retrieve the text entered by the user in a Streamlit interface.

    Returns:
        Dict[str, str]: A dictionary containing the user‑provided text under the
        ``"text"`` key. If the user has not entered anything, the value is an
        empty string.
    """
    user_input: str = st.text_input(label="Enter text", value="")
    return {"text": user_input}


def get_user_audio() -> bytes:
    """Capture audio from the user's microphone via Streamlit.

    Returns:
        bytes: Concatenated raw audio bytes captured from the microphone.

    Raises:
        RuntimeError: If the required Streamlit‑WebRTC component cannot be loaded,
            if the user does not provide any audio, or if an unexpected error occurs.
    """

    class _AudioRecorder(AudioProcessorBase):
        """Collect audio frames into a list for later concatenation."""

        def __init__(self) -> None:
            self._frames: List[bytes] = []

        def recv(self, frame):  # type: ignore[override]
            """Receive an audio frame and store its raw bytes."""
            ndarray = frame.to_ndarray(format="s16")
            self._frames.append(ndarray.tobytes())
            return frame

        def get_audio_bytes(self) -> bytes:
            """Concatenate all stored frames into a single ``bytes`` object."""
            return b"".join(self._frames)

    webrtc_ctx = webrtc_streamer(
        key="audio_recorder",
        mode=WebRtcMode.SENDRECV,
        audio_processor_factory=_AudioRecorder,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=False,
    )

    if not webrtc_ctx.state.playing:
        st.info("Cliquez sur le bouton **Start** ci‑dessus pour commencer l'enregistrement audio.")
        raise RuntimeError("Aucun enregistrement audio n'est en cours.")

    if webrtc_ctx.state == webrtc_ctx.state.STOPPED:
        processor = webrtc_ctx.audio_processor
        if processor is None:
            raise RuntimeError("Le processeur audio n'est pas disponible.")
        audio_bytes = processor.get_audio_bytes()
        if not audio_bytes:
            raise RuntimeError("Aucun audio n'a été capturé.")
        return audio_bytes

    raise RuntimeError("L'enregistrement audio est toujours en cours. Veuillez l'arrêter avant de récupérer les données.")


def play_audio(audio_bytes: bytes) -> bool:
    """Play synthesized speech audio in a Streamlit interface.

    Args:
        audio_bytes (bytes): Audio data to be played. Must be a non‑empty byte string.

    Returns:
        bool: ``True`` if the audio was successfully sent to Streamlit for playback.

    Raises:
        RuntimeError: If ``audio_bytes`` is ``None`` or empty.
    """
    if not audio_bytes:
        raise RuntimeError("Le paramètre audio_bytes est manquant ou vide.")
    try:
        st.audio(audio_bytes, format="audio/wav")
    except Exception as exc:
        raise RuntimeError(f"Erreur lors de la lecture de l'audio : {exc}")
    return True


def update_conversation_history(
    history: List[Dict[str, Any]], role: str, content: str
) -> List[Dict[str, Any]]:
    """Append a new message to the conversation history.

    Args:
        history: Existing conversation history.
        role: Role of the new message (e.g., "user" or "assistant").
        content: Text content of the new message.

    Returns:
        Updated conversation history with the new message appended.

    Raises:
        RuntimeError: If inputs are of incorrect types.
    """
    if not isinstance(history, list):
        raise RuntimeError("Le paramètre 'history' doit être une liste.")
    for idx, entry in enumerate(history):
        if not isinstance(entry, dict):
            raise RuntimeError(f"L'élément d'index {idx} dans 'history' doit être un dictionnaire.")
    if not isinstance(role, str):
        raise RuntimeError("Le paramètre 'role' doit être une chaîne de caractères.")
    if not isinstance(content, str):
        raise RuntimeError("Le paramètre 'content' doit être une chaîne de caractères.")
    updated_history = history.copy()
    updated_history.append({"role": role, "content": content})
    return updated_history


# --------------------------------------------------------------------------- #
#                               TEXT‑TO‑SPEECH (FILE)                        #
# --------------------------------------------------------------------------- #

SUPPORTED_VOICES = [
    "Aaliyah-PlayAI",
    "Adelaide-PlayAI",
    "Angelo-PlayAI",
    "Arista-PlayAI",
    "Atlas-PlayAI",
    "Basil-PlayAI",
    "Briggs-PlayAI",
    "Calum-PlayAI",
    "Celeste-PlayAI",
    "Cheyenne-PlayAI",
    "Chip-PlayAI",
    "Cillian-PlayAI",
    "Deedee-PlayAI",
    "Eleanor-PlayAI",
    "Fritz-PlayAI",
    "Gail-PlayAI",
    "Indigo-PlayAI",
    "Jennifer-PlayAI",
    "Judy-PlayAI",
    "Mamaw-PlayAI",
    "Mason-PlayAI",
    "Mikail-PlayAI",
    "Mitch-PlayAI",
    "Nia-PlayAI",
    "Quinn-PlayAI",
    "Ruby-PlayAI",
    "Thunder-PlayAI",
]

_ALLOWED_FORMATS = {"mp3", "opus", "aac", "flac", "wav"}


def generate_speech(
    text: str,
    voice: str = "Aaliyah-PlayAI",
    speed: float = 1.0,
    output_format: str = "mp3",
    output_dir: str | os.PathLike = ".",
) -> Dict[str, str | int]:
    """Generate an audio file from *text* using the PlayAI TTS model."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("`text` must be a non‑empty string.")
    if voice not in SUPPORTED_VOICES:
        raise ValueError(f"`voice` must be one of the supported voices: {SUPPORTED_VOICES}")
    if not isinstance(speed, (int, float)):
        raise ValueError("`speed` must be a numeric value.")
    if not 0.25 <= float(speed) <= 4.0:
        raise ValueError("`speed` must be between 0.25 and 4.0 (inclusive).")
    if output_format not in _ALLOWED_FORMATS:
        raise ValueError(f"`output_format` must be one of {_ALLOWED_FORMATS}, got '{output_format}'.")
    output_dir_path = Path(output_dir).expanduser().resolve()
    if not output_dir_path.is_dir():
        raise ValueError(f"`output_dir` must be an existing directory: {output_dir_path}")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Environment variable `GROQ_API_KEY` is not set.")

    client = Groq(api_key=api_key)

    try:
        response = client.audio.speech.create(
            model="playai-tts",
            input=text,
            voice=voice,
            speed=float(speed),
            response_format=output_format,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to generate speech via Groq API: {exc}") from exc

    unique_name = f"speech_{uuid.uuid4().hex}.{output_format}"
    audio_file_path = output_dir_path / unique_name

    try:
        response.write_to_file(str(audio_file_path))
    except Exception as exc:
        raise RuntimeError(f"Unable to write audio file to disk: {exc}") from exc

    return {
        "audio_file_path": str(audio_file_path),
        "format": output_format,
        "text_length": len(text),
        "voice_used": voice,
    }


def generate_jarvis_reply(history: List[Dict[str, str]]) -> Dict[str, Any]:
    """Generate a Jarvis‑style reply based on conversation history."""
    if not isinstance(history, list):
        raise ValueError("`history` must be a list of message dictionaries.")
    if len(history) == 0:
        raise ValueError("`history` cannot be empty; at least one user message is required.")
    for idx, msg in enumerate(history):
        if not isinstance(msg, dict):
            raise ValueError(f"Message at index {idx} is not a dict.")
        if "role" not in msg or "content" not in msg:
            raise ValueError(f"Message at index {idx} must contain 'role' and 'content' keys.")
        if msg["role"] not in {"user", "assistant"}:
            raise ValueError(f"Message at index {idx} has invalid role '{msg['role']}'.")
        if not isinstance(msg["content"], str) or not msg["content"].strip():
            raise ValueError(f"Message at index {idx} has empty or non‑string content.")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Environment variable `GROQ_API_KEY` is not set.")

    try:
        groq_client = Groq(api_key=api_key)
    except Exception as exc:
        raise RuntimeError(f"Failed to initialise Groq client: {exc}") from exc

    system_message = {
        "role": "system",
        "content": (
            "You are Jarvis, an AI assistant that replies in a helpful, concise, "
            "and friendly manner. Use the provided conversation history to "
            "understand context and answer the latest user query."
        ),
    }

    messages: List[Dict[str, str]] = [system_message] + history

    try:
        llm_response = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            temperature=0.5,
            max_tokens=1024,
            top_p=1.0,
            stream=False,
        )
    except Exception as exc:
        raise RuntimeError(f"Groq API request failed: {exc}") from exc

    try:
        reply_text: str = llm_response.choices[0].message.content  # type: ignore
    except (AttributeError, IndexError) as exc:
        raise RuntimeError("Unexpected response format from Groq API.") from exc

    return {"reply": reply_text}


# --------------------------------------------------------------------------- #
#                               TEXT‑TO‑SPEECH (BYTES)                       #
# --------------------------------------------------------------------------- #

SUPPORTED_FORMATS = {"mp3", "opus", "aac", "flac", "wav"}


class GroqTTSConfigurationError(RuntimeError):
    """Raised when the environment or caller configuration is invalid."""


def generate_speech_bytes(
    *,
    text: str,
    voice: str = "Aaliyah-PlayAI",
    speed: float = 1.0,
    response_format: str = "mp3",
) -> Dict[str, bytes]:
    """Convert *text* to spoken audio using Groq’s PlayAI TTS model."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("`text` must be a non‑empty string.")
    if voice not in SUPPORTED_VOICES:
        raise ValueError(f"`voice` must be one of {SUPPORTED_VOICES!r}. Received: {voice!r}")
    if not isinstance(speed, (int, float)):
        raise ValueError("`speed` must be a numeric type.")
    if not 0.25 <= speed <= 4.0:
        raise ValueError("`speed` must be between 0.25 and 4.0 (inclusive).")
    if response_format not in SUPPORTED_FORMATS:
        raise ValueError(f"`response_format` must be one of {sorted(SUPPORTED_FORMATS)}.")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise GroqTTSConfigurationError("Environment variable `GROQ_API_KEY` is not set.")

    client = Groq(api_key=api_key)

    try:
        response = client.audio.speech.create(
            model="playai-tts",
            input=text,
            voice=voice,
            speed=speed,
            response_format=response_format,
        )
    except Exception as exc:
        raise RuntimeError(f"Groq TTS request failed: {exc}")

    try:
        audio_bytes = response.content  # type: ignore[attr-defined]
    except Exception as exc:
        raise RuntimeError(f"Failed to retrieve audio bytes from response: {exc}")

    return {"audio_bytes": audio_bytes}


if __name__ == "__main__":
    print("Running my_agent...")
    # Placeholder for main workflow implementation.
    # Available functions:
    # - get_user_text()
    # - get_user_audio()
    # - play_audio()
    # - update_conversation_history()
    # - generate_speech()
    # - generate_jarvis_reply()
    # - generate_speech_bytes()
    pass