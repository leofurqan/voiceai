# Voice AI – Real-Time Speech-to-Speech Chat with OpenAI

A **full-duplex voice assistant** built with **Django Channels + WebSockets**, **OpenAI Whisper**, **GPT-4o-mini**, and **OpenAI TTS** (`tts-1-hd` with MP3 streaming).

- Speak → Transcribed → AI responds → Spoken back (female voice: `nova` / `shimmer`)
- Real-time **Voice Activity Detection (VAD)** with interrupt support
- Works on **desktop & mobile** (Chrome/Edge/Firefox)
- Optimized for **Pakistan (PK)** – low latency, clear audio

---

## Features

| Feature | Status |
|-------|--------|
| Real-time mic capture (16kHz PCM) | Done |
| WebRTC VAD + volume threshold | Done |
| Whisper STT (`whisper-1`) | Done |
| GPT-4o-mini chat | Done |
| OpenAI TTS streaming (`tts-1-hd`, MP3) | Done |
| Female voice (`nova` or `shimmer`) | Done |
| Interrupt AI while speaking | Done |
| Async WebSocket (Django Channels) | Done |
| Docker-ready (optional) | Done |

---

## Tech Stack

- **Backend**: Django 5.2 + Channels + ASGI
- **Realtime**: WebSockets + Redis Channel Layer
- **AI**: OpenAI (`whisper-1`, `gpt-4o-mini`, `tts-1-hd`)
- **Audio**: `pydub`, `webrtcvad`, `numpy`
- **Frontend**: Vanilla HTML + Web Audio API + MP3 Blob playback

---

## Prerequisites

| Tool | Version |
|------|--------|
| Python | 3.10+ |
| Redis | 6.0+ (running on port `31382`) |
| OpenAI API Key | Valid key with access to `whisper-1`, `gpt-4o-mini`, `tts-1-hd` |

---

## Create Virtual Environment
python -m venv venv
source venv/bin/activate    # Linux/Mac
# or
venv\Scripts\activate       # Windows

## Install Dependencies
pip install -r requirements.txt

## Create .env File
DEBUG=True
SECRET_KEY=your-super-secret-key-here
ALLOWED_HOSTS=127.0.0.1,localhost
OPENAI_API_KEY=sk-XXXXXXXXXXXXXXXXXXXXXXXX

## Start Redis (Docker or Local)
docker run -d --name redis-voice -p 31382:6379 redis:7

if your redis is on different port change it in settings.py

## Project Structure
voiceai/
├── chat/
│   ├── consumers.py     ← WebSocket logic
│   ├── routing.py
│   └── views.py
├── voiceai/
│   ├── settings.py
│   ├── asgi.py
│   └── urls.py
├── templates/
│   └── index.html       ← Frontend
├── requirements.txt
├── .env                 ← Your secrets
└── manage.py
