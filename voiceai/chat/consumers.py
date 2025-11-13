# chat/consumers.py
import json
import asyncio
import io
import logging
import numpy as np
from channels.generic.websocket import AsyncWebsocketConsumer
from pydub import AudioSegment
import webrtcvad
from openai import OpenAI
from decouple import config

logger = logging.getLogger(__name__)

class VoiceConsumer(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vad = webrtcvad.Vad(3)
        self.sample_rate = 16000
        self.frame_ms = 30
        self.frame_size = int(self.sample_rate * self.frame_ms / 1000) * 2  # 960 bytes
        self.audio_buf = b""
        self.speech_buf = b""
        self.is_speaking = False
        self.silence_frames = 0
        self.rms_history = []
        self.speaking_task = None
        self.client = OpenAI(api_key=config("OPENAI_API_KEY"))

    async def connect(self):
        await self.accept()
        await self.send(text_data=json.dumps({"type": "connected"}))
        logger.info("WebSocket connected")

    async def disconnect(self, code):
        if self.speaking_task:
            self.speaking_task.cancel()
        logger.info(f"Disconnected: {code}")

    async def receive(self, text_data=None, bytes_data=None):
        if not bytes_data:
            return

        self.audio_buf += bytes_data
        while len(self.audio_buf) >= self.frame_size:
            frame = self.audio_buf[:self.frame_size]
            self.audio_buf = self.audio_buf[self.frame_size:]

            if len(frame) < self.frame_size:
                continue

            # --- VAD: Volume + WebRTC ---
            audio_np = np.frombuffer(frame, dtype=np.int16)
            if len(audio_np) == 0:
                rms = 0.0
            else:
                squared = audio_np.astype(np.float64) ** 2
                mean_sq = np.mean(squared)
                rms = np.sqrt(mean_sq) if mean_sq > 0 else 0.0

            self.rms_history = (self.rms_history + [rms])[-20:]
            avg_rms = np.mean(self.rms_history) if self.rms_history else 0.0
            threshold = max(1500, avg_rms * 0.8)
            is_speech = rms > threshold and self.vad.is_speech(frame, self.sample_rate)

            await self.send(text_data=json.dumps({
                "type": "vad",
                "speech": bool(is_speech),
                "volume": int(rms) if not np.isnan(rms) else 0,
                "threshold": int(threshold),
                "avg": int(avg_rms) if not np.isnan(avg_rms) else 0
            }))

            # --- Interrupt AI ---
            if is_speech and rms > threshold * 1.2 and self.speaking_task:
                logger.info("INTERRUPT: User speaking")
                self.speaking_task.cancel()
                self.speaking_task = None
                await self.send(text_data=json.dumps({"type": "interrupt"}))

            # --- Accumulate speech ---
            if is_speech:
                self.silence_frames = 0
                self.speech_buf += frame
                self.is_speaking = True
            elif self.is_speaking:
                self.silence_frames += 1
                self.speech_buf += frame
                if self.silence_frames > 20:
                    self.is_speaking = False
                    await self.handle_speech(self.speech_buf)
                    self.speech_buf = b""

    async def handle_speech(self, audio_bytes: bytes):
        logger.info(f"Processing {len(audio_bytes)} bytes")
        try:
            # --- Whisper STT ---
            seg = AudioSegment(data=audio_bytes, sample_width=2, frame_rate=16000, channels=1)
            buf = io.BytesIO()
            seg.export(buf, format="wav")
            buf.seek(0)

            text = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=("speech.wav", buf, "audio/wav")
                ).text.strip()
            )
            await self.send(text_data=json.dumps({"type": "transcript", "text": text}))
            logger.info(f"User: {text}")

            # --- GPT-4o-mini ---
            chat_resp = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": text}],
                    max_tokens=80,
                    temperature=0.7
                )
            )
            ai_text = chat_resp.choices[0].message.content.strip()
            await self.send(text_data=json.dumps({"type": "ai_text", "text": ai_text}))

            # --- Stream OpenAI TTS ---
            self.speaking_task = asyncio.create_task(self.stream_openai_tts(ai_text))

        except Exception as e:
            logger.exception("Speech processing failed")
            await self.send(text_data=json.dumps({"type": "error", "text": str(e)}))

    async def stream_openai_tts(self, text: str):
        try:
            await self.send(text_data=json.dumps({"type": "start_audio"}))
            logger.info("Streaming OpenAI TTS...")

            with self.client.audio.speech.with_streaming_response.create(
                model="tts-1",
                voice="alloy",
                input=text
            ) as response:
                async for chunk in response.iter_bytes(chunk_size=1024):
                    if not self.speaking_task:
                        break
                    await self.send(bytes_data=chunk)
                    await asyncio.sleep(0.01)

            await self.send(text_data=json.dumps({"type": "end_audio"}))
            logger.info("TTS finished")

        except asyncio.CancelledError:
            logger.info("TTS interrupted")
        except Exception as e:
            logger.error(f"TTS error: {e}")
        finally:
            self.speaking_task = None