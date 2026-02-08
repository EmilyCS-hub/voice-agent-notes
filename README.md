# Voice Agent Notes (CLI)

Minimal end-to-end CLI that records a short voice note, transcribes it, summarizes the notes, and plays a TTS response. All models use OpenAI.

## Setup

Clone the repository and create / activate the project virtual environment. These commands assume macOS / zsh and Python 3.12.4 as used in this project.

```bash
git clone https://github.com/EmilyCS-hub/voice-agent-notes.git
cd voice-agent-notes

# create a venv using the project's helper (uses Python 3.12.4 in this repo)
uv venv --python=3.12.4 .venv

# activate the virtual environment
source .venv/bin/activate
```

1. Install dependencies:

If you use the `uv` helper in this project:

```bash
uv sync
```

Or using pip directly (editable install):

```bash
pip install -e .
```

2. Create .env.local and set your API key in .env.local
   
```bash
cp .env.example .env.local
```

```bash
LIVEKIT_URL="wss://your-livekit-host"
LIVEKIT_API_KEY="your_livekit_key"
LIVEKIT_API_SECRET="your_livekit_secret"
OPENAI_API_KEY="your_openai_key"
```

If you plan to use the LiveKit-based voice agents (`voice_assistant.py` and `realtime_voice_assistant.py`) you will need a LiveKit server (or LiveKit Cloud project) and API credentials. LiveKit is an open-source WebRTC platform — see https://livekit.io/ for hosted options, deployment guides, and admin UI to create API keys. Put the WebSocket URL and API key/secret into `.env.local` as shown above.

## Run

Here has two solutions: 
one is using generic OpenAI Agent.
another is using livekit for voice agent (STT-LLM-TTS and gpt-realtime)

### OpenAI Agent

On macOS, allow microphone access for your terminal app.

```bash
uv run main.py --seconds 12
```

### LiveKit Agent

### LiveKit STT-LLM-TTS
There is a LiveKit Agents worker entrypoint in `voice_assistant.py` that performs the same note-taking flow using a LiveKit room.

Run with the LiveKit Agents CLI:

```bash
uv run voice_assistant.py console
```

or run in dev mode

```bash

uv run voice_assistant.py dev
```

### Download required runtime files (ONNX models)

When running the LiveKit agent you may see an error like:

```
Could not find file "model_q8.onnx"
```

This means the turn-detector / plugin runtime files aren't present locally. The project includes a helper to download those files into the right place. Run one of the following from the project root (use whichever matches how you normally run Python in this repo):

```bash
# If your virtualenv is activated (recommended):
python voice_assistant.py download-files

# Or using the `uv` runner you may already use:
uv run voice_assistant.py download-files
```

After the download finishes, re-run the agent (for example `uv run voice_assistant.py console`) and the ONNX-related error should be resolved.

## LiveKit Realtime Agent

There is a realtime OpenAI agent worker in `realtime_voice_assistant.py` that loads instructions from
`instructions/realtime_voice_instruction.py`. To adopt different agent, just create new instruction py file and have `instruction_text = ''' ...''' ` instruction. 

Run it:


at command line:

```bash
uv run realtime_voice_assistant.py console
```

or

```bash
uv run realtime_voice_assistant.py dev
```

### Options

- `--seconds`: recording duration
- `--samplerate`: recording sample rate
- `--stt-model`: transcription model (default `gpt-4o-mini-transcribe`)
- `--summary-model`: summary model (default `gpt-4o-mini`)
- `--tts-model`: TTS model (default `gpt-4o-mini-tts`)
- `--voice`: TTS voice (default `alloy`)
- `--instruction`: system instruction for the summary step
- `--greeting`: greeting text spoken before recording
- `--followup`: prompt spoken after each summary
- `--closing`: closing text spoken when the session ends
- `--done-phrases`: comma-separated phrases that end the session

## Notes

- TTS output is generated as WAV and played back locally.
- If you get device errors, check your system audio input/output settings.
