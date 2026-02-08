import argparse
import os
import tempfile

import numpy as np
import sounddevice as sd
import soundfile as sf
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv(".env.local")

from instructions import voice_notes_instruction as instruction_module
SUMMARY_INSTRUCTION = instruction_module.SUMMARY_INSTRUCTION
GREETING_TEXT = instruction_module.GREETING_TEXT
FOLLOWUP_PROMPT = instruction_module.FOLLOWUP_PROMPT
CLOSING_TEXT = instruction_module.CLOSING_TEXT
DONE_PHRASES = instruction_module.DONE_PHRASES


def record_audio(seconds: int, samplerate: int = 16000) -> np.ndarray:
    if seconds <= 0:
        raise ValueError("Recording duration must be greater than zero")

    recording = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=1, dtype="float32")
    sd.wait()
    return recording.squeeze()


def save_wav(samples: np.ndarray, samplerate: int) -> str:
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, samples, samplerate)
    return path


def transcribe_audio(client: OpenAI, wav_path: str, model: str) -> str:
    with open(wav_path, "rb") as audio_file:
        result = client.audio.transcriptions.create(model=model, file=audio_file)
    return result.text.strip()


def summarize_text(client: OpenAI, text: str, model: str, instruction: str) -> str:
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": instruction,
            },
            {"role": "user", "content": text},
        ],
    )
    return response.output_text.strip()


def synthesize_speech(client: OpenAI, text: str, model: str, voice: str) -> str:
    response = client.audio.speech.create(
        model=model,
        voice=voice,
        input=text,
        response_format="wav",
    )
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    response.stream_to_file(path)
    return path


def play_wav(path: str) -> None:
    data, samplerate = sf.read(path, dtype="float32")
    sd.play(data, samplerate)
    sd.wait()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record voice notes, summarize, and play TTS.")
    parser.add_argument("--seconds", type=int, default=10, help="Recording duration in seconds")
    parser.add_argument("--samplerate", type=int, default=16000, help="Recording sample rate")
    parser.add_argument("--stt-model", default="gpt-4o-mini-transcribe", help="OpenAI STT model")
    parser.add_argument("--summary-model", default="gpt-4o-mini", help="OpenAI model for summary")
    parser.add_argument("--tts-model", default="gpt-4o-mini-tts", help="OpenAI TTS model")
    parser.add_argument("--voice", default="alloy", help="TTS voice")
    parser.add_argument(
        "--instruction",
        default=SUMMARY_INSTRUCTION,
        help="System instruction for the summary step",
    )
    parser.add_argument(
        "--greeting",
        default=GREETING_TEXT,
        help="Greeting text spoken before recording",
    )
    parser.add_argument(
        "--followup",
        default=FOLLOWUP_PROMPT,
        help="Prompt spoken after each summary",
    )
    parser.add_argument(
        "--closing",
        default=CLOSING_TEXT,
        help="Closing text spoken when the session ends",
    )
    parser.add_argument(
        "--done-phrases",
        default=", ".join(DONE_PHRASES),
        help="Comma-separated phrases that end the session",
    )
    return parser.parse_args()


def is_done(transcript: str, done_phrases: list[str]) -> bool:
    text = transcript.strip().lower()
    return any(phrase in text for phrase in done_phrases)


def main() -> None:
    args = parse_args()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=api_key)

    if args.greeting:
        print("Saying greeting...")
        hello_path = synthesize_speech(client, args.greeting, model=args.tts_model, voice=args.voice)
        play_wav(hello_path)

    done_phrases = [phrase.strip().lower() for phrase in args.done_phrases.split(",") if phrase.strip()]

    while True:
        print(f"Recording for {args.seconds} seconds...")
        samples = record_audio(args.seconds, samplerate=args.samplerate)
        wav_path = save_wav(samples, samplerate=args.samplerate)

        print("Transcribing...")
        transcript = transcribe_audio(client, wav_path, args.stt_model)
        if not transcript:
            print("No transcription returned; try again.")
            continue

        print("Transcript:")
        print(transcript)

        if done_phrases and is_done(transcript, done_phrases):
            print("Closing...")
            closing_path = synthesize_speech(client, args.closing, model=args.tts_model, voice=args.voice)
            play_wav(closing_path)
            break

        print("Summarizing...")
        summary = summarize_text(client, transcript, args.summary_model, args.instruction)
        print("Summary:")
        print(summary)

        print("Generating TTS...")
        tts_path = synthesize_speech(client, summary, model=args.tts_model, voice=args.voice)

        print("Playing TTS...")
        play_wav(tts_path)

        if args.followup:
            print("Asking follow-up...")
            followup_path = synthesize_speech(
                client,
                args.followup,
                model=args.tts_model,
                voice=args.voice,
            )
            play_wav(followup_path)


if __name__ == "__main__":
    main()
