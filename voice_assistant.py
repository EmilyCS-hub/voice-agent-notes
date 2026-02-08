# uv run api_service/libs/services/voice_agent_livekit.py console
import logging

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.plugins import noise_cancellation, silero, openai
# Turn detector import removed: not required because OpenAI STT handles language detection
# from livekit.plugins.turn_detector.multilingual import MultilingualModel

import instructions.realtime_voice_instruction as instructionlib

import os

logger = logging.getLogger(__name__)
load_dotenv(".env.local")

# Basic logging configuration for standalone runs
logging.basicConfig(level=logging.INFO)

# Read important environment variables once and warn early if missing
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not set. OpenAI-based STT/LLM/TTS may fail at runtime.")


class Assistant(Agent):
    def __init__(self, instructions: str = "") -> None:
        super().__init__(
            instructions=instructions,
            stt=openai.STT(
                model="gpt-4o-mini-transcribe",  # "whisper"
                api_key=OPENAI_API_KEY,
            ),
            llm=openai.LLM(
                model="gpt-4.1-mini",
                api_key=OPENAI_API_KEY,
            ),
            tts=openai.TTS(
                model="gpt-4o-mini-tts",
                api_key=OPENAI_API_KEY,
                voice="alloy",
            ),
            # turn_detection=MultilingualModel(),
        )

    async def on_enter(self):
        # The agent should be polite and greet the user when it joins :)
        # session may not always be available during unit tests or certain lifecycle events
        if not getattr(self, "session", None):
            logger.warning("Agent session not available in on_enter; skipping greeting")
            return

        # generate_reply is an async operation in the LiveKit agent API; await it
        try:
            await self.session.generate_reply(
                user_input="Hello",
                allow_interruptions=True,
            )
        except Exception:
            logger.exception("Failed to generate greeting reply in on_enter")

    # To add tools, use the @function_tool decorator.
    # Here's an example that adds a simple weather tool.
    # You also have to add `from livekit.agents import function_tool, RunContext` to the top of this file
    # @function_tool
    # async def lookup_weather(self, context: RunContext, location: str):
    #     """Use this tool to look up current weather information in the given location.
    #
    #     If the location is not supported by the weather service, the tool will indicate this. You must tell the user the location's weather is unavailable.
    #
    #     Args:
    #         location: The location to look up weather information for (e.g. city name)
    #     """
    #
    #     logger.info(f"Looking up weather for {location}")
    #
    #     return "sunny with a temperature of 70 degrees."


def prewarm(proc: JobProcess):
    try:
        proc.userdata["vad"] = silero.VAD.load()
        logger.info("Silero VAD prewarmed successfully")
    except Exception:
        logger.exception("Failed to prewarm Silero VAD")
        # store a sentinel so callers can detect failure
        proc.userdata["vad"] = None


async def entrypoint(ctx: JobContext, instructions: str = ""):
    # Logging setup
    # Add any other context you want in all log entries here

    ctx.log_context_fields = {
        "room": ctx.room.name,
    }
    logger.info(f"connecting to room {ctx.room.name}")
    # participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant")

    # Set up a voice AI pipeline using OpenAI, Cartesia, AssemblyAI, and the LiveKit turn detector
    # session = AgentSession(
    #     # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
    #     # See all available models at https://docs.livekit.io/agents/models/stt/
    #     stt="assemblyai/universal-streaming:en",
    #     # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
    #     # See all available models at https://docs.livekit.io/agents/models/llm/
    #     llm="openai/gpt-4.1-mini",
    #     # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
    #     # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
    #     tts="cartesia/sonic-2:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
    #     # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
    #     # See more at https://docs.livekit.io/agents/build/turns
    #     turn_detection=MultilingualModel(),
    #     vad=ctx.proc.userdata["vad"],
    #     # allow the LLM to generate a response while waiting for the end of turn
    #     # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
    #     preemptive_generation=True,
    # )
    # Use the prewarmed VAD if available; log and continue if not
    vad = ctx.proc.userdata.get("vad")
    if vad is None:
        logger.warning("VAD not available from prewarm; continuing without VAD (turn detection may be affected)")

    session = AgentSession(
        vad=vad,
        # minimum delay for endpointing, used when turn detector believes the user is done with their turn
        min_endpointing_delay=0.5,
        # maximum delay for endpointing, used when turn detector does not believe the user is done with their turn
        max_endpointing_delay=5.0,
        preemptive_generation=True,
    )

    # Metrics collection, to measure pipeline performance
    # For more information, see https://docs.livekit.io/agents/build/metrics/
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)
        print(f"Metrics collected: {ev.metrics}")

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    try:
        await session.start(
            agent=Assistant(instructions=instructionlib.instruction_text),
            room=ctx.room,
            room_input_options=RoomInputOptions(
                # For telephony applications, use `BVCTelephony` for best results
                noise_cancellation=noise_cancellation.BVC(),
            ),
        )
    except Exception:
        logger.exception("Failed to start AgentSession")
        # attempt a best-effort shutdown
        try:
            if hasattr(session, "stop"):
                await session.stop()
        except Exception:
            logger.exception("Error while stopping session after failed start")
        # re-raise to ensure the worker knows the entrypoint failed
        raise
    finally:
        logger.info("Agent entrypoint finished (session lifecycle complete or failed)")

    # Join the room and connect to the user
    # await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
