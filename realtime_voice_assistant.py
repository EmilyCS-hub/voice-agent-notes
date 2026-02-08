# uv run api_service/libs/services/realtime_agent.py console
import logging

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    RoomOutputOptions,
    WorkerOptions,
    cli,
    metrics,
    AutoSubscribe,
    UserInputTranscribedEvent,
    ConversationItemAddedEvent,
)
from livekit.agents.llm import ImageContent, AudioContent
from livekit.plugins import noise_cancellation, silero, openai
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.agents import ChatContext, ChatMessage
from datetime import datetime
import json
import os

logger = logging.getLogger("agent")
load_dotenv(".env.local")

import instructions.realtime_voice_instruction as instructionlib


class Assistant(Agent):
    def __init__(self, instructions: str = "") -> None:
        super().__init__(
            instructions=instructions,
            llm=openai.realtime.RealtimeModel(
                model="gpt-realtime-mini",
                api_key=os.getenv("OPENAI_API_KEY"),
                voice="alloy",
            ),
        )

    async def on_enter(self):
        # The agent should be polite and greet the user when it joins :)
        self.session.generate_reply(
            user_input="Hello",
            allow_interruptions=True,
        )

    async def on_exit(self):
        await self.session.generate_reply(
            instructions="Tell the user a friendly goodbye before you exit.",
        )

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
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext, instructions: str = ""):
    # Logging setup

    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    logger.info(f"connecting to room {ctx.room.name}")
    # participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant")

    # ---- Retrieve metadata HERE ----
    raw_meta = ctx.job.metadata or ctx.room.metadata or {}
    logger.info(f"Raw metadata: {raw_meta}")
    logger.info(f"ctx.job.metadata: {ctx.job.metadata}")
    logger.info(f"ctx.room.metadata: {ctx.room.metadata}")

    try:
        metadata = json.loads(raw_meta) if isinstance(raw_meta, str) else raw_meta
    except Exception as e:
        logger.error(f"Error parsing metadata: {e}")
        metadata = {}

    agent_name = metadata.get("agentName")
    logger.info(f"Agent selected: {agent_name}")
    # ----------------------------------

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
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        # minimum delay for endpointing, used when turn detector believes the user is done with their turn
        min_endpointing_delay=0.5,
        # maximum delay for endpointing, used when turn detector does not believe the user is done with their turn
        max_endpointing_delay=5.0,
        preemptive_generation=True,
        use_tts_aligned_transcript=True,
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

    async def write_transcript():
        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"Writing transcript for session at {current_date}")
        logger.info(session.history.to_dict())
        logger.info(f"write_transcript() completed.")

    ctx.add_shutdown_callback(log_usage)
    ctx.add_shutdown_callback(write_transcript)

    @session.on("close")
    def on_session_close():
        print("Session is closing, writing final transcript...")

    @session.on("user_input_transcribed")
    def on_user_input_transcribed(event: UserInputTranscribedEvent):
        print(
            f"Begin: 🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄"
            f"User input transcribed: {event.transcript}, "
            f"language: {event.language}, "
            f"final: {event.is_final}, "
            f"speaker id: {event.speaker_id}"
            f"End:🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄🎄"
        )

    @session.on("conversation_item_added")
    def on_conversation_item_added(event: ConversationItemAddedEvent):
        print(
            f"Begin: 🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖"
            f"Conversation item added from {event.item.role}: {event.item.text_content}. interrupted: {event.item.interrupted}"
        )
        # to iterate over all types of content:
        for content in event.item.content:
            if isinstance(content, str):
                print(f" - text: {content}")
            elif isinstance(content, ImageContent):
                # image is either a rtc.VideoFrame or URL to the image
                print(f" - image: {content.image}")
            elif isinstance(content, AudioContent):
                # frame is a list[rtc.AudioFrame]
                print(f" - audio: {content.frame}, transcript: {content.transcript}")
        print(f"End: 🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖")

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    isEnableVideo = os.getenv("ENABLE_VIDEO", "false").lower() == "true"
    await session.start(
        agent=Assistant(instructions=instructionlib.instruction_text),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
        room_output_options=RoomOutputOptions(sync_transcription=True),
        # video_enabled=isEnableVideo,
    )

    # Join the room and connect to the user
    if isEnableVideo:
        await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)
    else:
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
