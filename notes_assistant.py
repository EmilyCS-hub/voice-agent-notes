import asyncio
import os
from datetime import datetime

from livekit import agents
from livekit.agents import Agent, AgentSession, JobContext, RoomOutputOptions
from livekit.plugins import openai, silero
import instructions.realtime_voice_instruction as instructionlib

import instructions.voice_notes_instruction as instructionnotelib
from instructions.voice_notes_instruction import (
    CLOSING_TEXT,
    DONE_PHRASES,
    FOLLOWUP_PROMPT,
    GREETING_TEXT,
    SUMMARY_INSTRUCTION,
)


def build_agent_instructions() -> str:
    done_examples = ", ".join(DONE_PHRASES)
    return (
        f"{instructionlib.instruction_text}\n"
        f"After each response, ask: {FOLLOWUP_PROMPT}\n"
        "If the user indicates they are finished, respond with the closing text "
        f"and do not ask another question. Examples of done phrases: {done_examples}."
    )


class NotesAgent(Agent):
    def __init__(
        self, ctx: JobContext, stop_event: asyncio.Event, transcript: list[str]
    ):
        super().__init__(instructions=build_agent_instructions())
        self.ctx = ctx
        self.stop_event = stop_event
        self.transcript = transcript
        self.done_phrases = [phrase.lower() for phrase in DONE_PHRASES]
        # track pending transcript handler tasks so we can await them on shutdown
        self._pending_transcript_tasks: set[asyncio.Task] = set()

    async def on_enter(self):
        if GREETING_TEXT:
            self.session.say(GREETING_TEXT, add_to_chat_ctx=True)

        # ---- Transcript listener (sync wrapper, async inside) ----
        def handle_transcript(event):
            # schedule the transcript handler and keep track of the task so we don't lose data
            task = asyncio.create_task(self.on_transcript(event))
            self._pending_transcript_tasks.add(task)

            def _on_done(t: asyncio.Task):
                try:
                    self._pending_transcript_tasks.discard(t)
                except Exception:
                    pass

        self.session.on("transcript", handle_transcript)

        # Also listen for conversation items so we capture agent (assistant) messages
        def handle_conversation_item(event):
            # schedule conversation item handling similarly to transcript handling
            task = asyncio.create_task(self.on_conversation_item(event))
            self._pending_transcript_tasks.add(task)

            def _on_done_conv(t: asyncio.Task):
                try:
                    self._pending_transcript_tasks.discard(t)
                except Exception:
                    pass

            task.add_done_callback(_on_done_conv)

        try:
            self.session.on("conversation_item_added", handle_conversation_item)
        except Exception:
            # Some AgentSession implementations may not support this event
            print("Note: conversation_item_added event not supported by this session")
        print("📜 Transcript listener registered.")

    async def on_transcript(self, event):
        # Try to infer speaker/role from available event fields
        role = None
        if hasattr(event, "speaker_id") and getattr(event, "speaker_id"):
            role = "USER"
        elif hasattr(event, "role") and getattr(event, "role"):
            role = getattr(event, "role").upper()
        else:
            # Fall back to is_final heuristic (not ideal)
            role = "USER" if getattr(event, "is_final", False) else "AGENT"

        # Prefer common text fields
        text = getattr(event, "text", None) or getattr(event, "transcript", None) or getattr(event, "user_transcript", None) or ""
        line = f"[{role}] {text}"
        print(line)  # show in terminal
        self.transcript.append(line)  # store for file

    async def on_conversation_item(self, event):
        # conversation items can contain multiple types of content; extract text pieces
        try:
            item = getattr(event, "item", None)
            if not item:
                return

            role = getattr(item, "role", "AGENT").upper()
            # content may be a list of strings or content objects
            for content in getattr(item, "content", []) or []:
                if isinstance(content, str):
                    line = f"[{role}] {content}"
                    print(line)
                    self.transcript.append(line)
                else:
                    # some content types expose a transcript property
                    txt = getattr(content, "transcript", None)
                    if txt:
                        line = f"[{role}] {txt}"
                        print(line)
                        self.transcript.append(line)
        except Exception:
            print("Error handling conversation item event")

    async def on_user_turn_completed(self, turn_ctx, new_message):
        text = (new_message.text_content or "").lower()

        if any(phrase in text for phrase in self.done_phrases):
            # Say closing text (try/except because say() may return sync or async)
            speech = None
            try:
                speech = self.session.say(CLOSING_TEXT, add_to_chat_ctx=True)
                try:
                    await speech
                except TypeError:
                    # some implementations return non-awaitable
                    pass
            except Exception:
                print("Failed to play closing text")

            # trigger graceful shutdown for the entrypoint
            self.stop_event.set()

            # Attempt to stop the session and disconnect the room so the agent ends itself
            try:
                if getattr(self, "session", None) and hasattr(self.session, "stop"):
                    await self.session.stop()
            except Exception:
                print("Warning: failed to stop AgentSession from agent")

            try:
                if getattr(self, "ctx", None) and getattr(self.ctx, "room", None):
                    await self.ctx.room.disconnect()
            except Exception:
                print("Warning: failed to disconnect room from agent")

            # Wait for any pending transcript tasks to complete so save_notes captures them
            if self._pending_transcript_tasks:
                try:
                    await asyncio.gather(*list(self._pending_transcript_tasks), return_exceptions=True)
                except Exception:
                    print("Warning: error while awaiting pending transcript tasks")

async def save_notes(transcript: list[str]):
    # Save notes relative to this script so files are predictable regardless of CWD
    base_dir = os.path.dirname(os.path.abspath(__file__))
    notes_dir = os.path.join(base_dir, "notes")
    print(f"Saving notes to directory: {notes_dir}")
    os.makedirs(notes_dir, exist_ok=True)

    filename = datetime.now().strftime(os.path.join(notes_dir, "%Y-%m-%d_%H-%M-%S.txt"))

    # If transcript is empty, still create a short file explaining no content was captured
    if not transcript:
        contents = "(no transcript captured)"
    else:
        contents = "\n".join(transcript)

    with open(filename, "w", encoding="utf-8") as f:
        f.write(contents)

    print(f"💾 Notes saved to: {filename} (lines: {len(transcript)})")


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    stop_event = asyncio.Event()
    transcript: list[str] = []

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=openai.STT(model=os.getenv("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe")),
        llm=openai.LLM(model=os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")),
        tts=openai.TTS(
            model=os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts"),
            voice=os.getenv("OPENAI_TTS_VOICE", "alloy"),
        ),
    )

    # Register a shutdown callback to ensure notes are saved even on external termination
    async def _save_on_shutdown():
        print("Shutdown callback: saving notes...")
        try:
            await save_notes(transcript)
        except Exception:
            print("Error while saving notes in shutdown callback")

    try:
        ctx.add_shutdown_callback(_save_on_shutdown)
    except Exception:
        # Some older versions of the agents API may not provide add_shutdown_callback
        # In that case, save_notes will still run in the finally block below.
        pass

    await session.start(
        agent=NotesAgent(ctx, stop_event, transcript),
        room=ctx.room,
        room_output_options=RoomOutputOptions(sync_transcription=True),
    )

    async def on_disconnected(_):
        stop_event.set()

    ctx.room.on("disconnected", on_disconnected)

    try:
        await stop_event.wait()
    finally:
        print("🔄 Shutting down and saving notes...")
        await ctx.room.disconnect()
        await asyncio.sleep(0.2)  # give LiveKit time to clean up
        await save_notes(transcript)


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
