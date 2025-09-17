import asyncio
import sys
import logging
from dotenv import load_dotenv
from openai import OpenAI
from quart import Quart, request, jsonify
from quart_cors import cors
from computer import Computer
from agent import ComputerAgent

# Configure logging
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()
client = OpenAI()

# TODO this should be a system prompt
USER_PROMPT = """
The user's host operating system is macOS.
- Open applications with Spotlight (cmd+space)
- See open windows and workspaces with Mission Control (ctrl+up arrow)
""".strip()

app = Quart(__name__)
app = cors(app, allow_origin="*", allow_methods=["POST", "OPTIONS"], allow_headers=["Content-Type"])

# Global computer and agent instances
computer = None
agent = None

async def initialize_agent():
    global computer, agent
    logger.info("Initializing agent and computer...")

    if computer is None:
        logger.info("Creating Computer instance...")
        computer = Computer(use_host_computer_server=True)
        await computer.__aenter__()
        logger.info("Computer initialized successfully")

    if agent is None:
        logger.info("Creating ComputerAgent instance...")
        agent = ComputerAgent(
            model="anthropic/claude-sonnet-4-20250514",
            tools=[computer],
            max_trajectory_budget=5.0,
            use_prompt_caching=True,
            only_n_most_recent_images=2,
        )
        logger.info("Agent initialized successfully")

@app.route('/process', methods=['POST'])
async def process_input():
    request_id = id(request)
    logger.info(f"[Request {request_id}] New audio processing request received")

    try:
        logger.info(f"[Request {request_id}] Initializing agent...")
        await initialize_agent()

        logger.info(f"[Request {request_id}] Parsing request files...")
        files = await request.files
        if 'audio' not in files:
            logger.error(f"[Request {request_id}] Missing audio file in request")
            return jsonify({'error': 'Missing audio file in request'}), 400

        audio_file = files['audio']
        logger.info(f"[Request {request_id}] Audio file received: {audio_file.filename}")

        if not audio_file.filename.endswith('.wav'):
            logger.error(f"[Request {request_id}] Invalid file format: {audio_file.filename}")
            return jsonify({'error': 'Only WAV files are supported'}), 400

        # Transcribe the audio using OpenAI
        logger.info(f"[Request {request_id}] Reading audio file content...")
        audio_content = audio_file.read()
        audio_size = len(audio_content)
        logger.info(f"[Request {request_id}] Audio file size: {audio_size} bytes ({audio_size/1024:.1f} KB)")

        # Create a BytesIO object for OpenAI client
        from io import BytesIO
        audio_buffer = BytesIO(audio_content)
        audio_buffer.name = audio_file.filename  # OpenAI client needs a filename

        logger.info(f"[Request {request_id}] Sending audio to OpenAI for transcription...")
        transcription_start = asyncio.get_event_loop().time()

        transcription = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=audio_buffer
        )

        transcription_duration = asyncio.get_event_loop().time() - transcription_start
        user_input = transcription.text
        print(f"Got transcription: {user_input}")
        logger.info(f"[Request {request_id}] Transcription completed in {transcription_duration:.2f}s")
        logger.info(f"[Request {request_id}] Transcribed text ({len(user_input)} chars): '{user_input}'")

        messages = [
            {"role": "user", "content": USER_PROMPT},
            {"role": "user", "content": user_input},
        ]

        logger.info(f"[Request {request_id}] Running agent with transcribed input...")
        results = []
        agent_start = asyncio.get_event_loop().time()

        async for result in agent.run(messages):
            logger.debug(f"[Request {request_id}] Agent result: {result}")
            for item in result['output']:
                if item['type'] == 'message':
                    message_text = item['content'][0]['text']
                    results.append(message_text)
                    print(message_text)
                    logger.info(f"[Request {request_id}] Agent message: {message_text}")

        agent_duration = asyncio.get_event_loop().time() - agent_start
        logger.info(f"[Request {request_id}] Agent processing completed in {agent_duration:.2f}s")
        logger.info(f"[Request {request_id}] Total results: {len(results)} messages")

        response_data = {
            'success': True,
            'transcription': user_input,
            'results': results
        }
        logger.info(f"[Request {request_id}] Request completed successfully")
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"[Request {request_id}] Error processing request: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

async def main():
    async with Computer(
        use_host_computer_server=True
    ) as computer:
        agent = ComputerAgent(
            model="anthropic/claude-sonnet-4-20250514",
            tools=[computer],
            max_trajectory_budget=5.0,
            use_prompt_caching=True,
            only_n_most_recent_images=2,
        )

        messages = [
            {"role": "user", "content": USER_PROMPT}, # TODO figure out if 'system' is correct for claude
            {"role": "user", "content": "Open firefox and go to github.com."},
        ]

        screenshots = [await computer.interface.screenshot()]

        async for result in agent.run(messages):
            print("RESULT:", result, end='\n\n', file=sys.stderr)
            for item in result['output']:
                if item['type'] == 'message':
                    print(item['content'][0]['text'])

        screenshots.append(await computer.interface.screenshot())

        for i, screenshot in enumerate(screenshots):
            with open(f'screenshot{i}.png', 'wb') as f:
                f.write(screenshot)

if __name__ == "__main__":
    # Run the Quart server instead of the main function
    logger.info("Starting Quart server on http://localhost:8002")
    logger.info("Server ready to accept audio processing requests at /process endpoint")
    app.run(host='localhost', port=8002, debug=True)
