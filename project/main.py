import asyncio
import sys
import logging
from io import BytesIO
from dotenv import load_dotenv
from openai import OpenAI
from quart import Quart, request, jsonify
from quart_cors import cors
from computer import Computer
from agent import ComputerAgent
import computer_server

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

# TODO this should be a system prompt
USER_PROMPT = """
The user's host operating system is macOS.
- Open applications with Spotlight (cmd+space)
- See open windows and workspaces with Mission Control (ctrl+up arrow)
""".strip()

app = Quart(__name__)
app = cors(app, allow_origin="*", allow_methods=["POST", "OPTIONS"], allow_headers=["Content-Type"])

openai = None
computer = None
agent = None

async def init_agent():
    global openai, computer, agent

    if openai is None:
        logger.info("Creating OpenAI instance...")
        openai = OpenAI()
        logger.info("OpenAI initialized successfully")

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

async def run_agent(user_input, request_id):
    """Common function to process text input through the agent."""
    messages = [
        {"role": "user", "content": USER_PROMPT},
        {"role": "user", "content": user_input},
    ]

    logger.info(f"[Request {request_id}] Running agent with input...")
    results = []
    agent_start = asyncio.get_event_loop().time()

    async for result in agent.run(messages):
        print("RESULT:", result, file=sys.stderr)
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

    return results

@app.route('/process', methods=['POST'])
async def process_input():
    request_id = id(request)
    logger.info(f"[Request {request_id}] New text processing request received")

    try:
        logger.info(f"[Request {request_id}] Parsing JSON request...")
        data = await request.get_json()

        if not data or 'text' not in data:
            logger.error(f"[Request {request_id}] Missing 'text' field in JSON body")
            return jsonify({'error': "Missing 'text' field in JSON body"}), 400

        user_input = data['text']
        logger.info(f"[Request {request_id}] Text input ({len(user_input)} chars): '{user_input}'")

        results = await run_agent(user_input, request_id)

        response_data = {
            'success': True,
            'results': results
        }
        logger.info(f"[Request {request_id}] Request completed successfully")
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"[Request {request_id}] Error processing request: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

# TODO this should probably be async?
def transcribe(audio_file):
    if not audio_file.filename.endswith(".wav"):
        raise Exception(f"Invalid filename: {audio_file.filename}")

    audio_content = audio_file.read()
    audio_size = len(audio_content)
    audio_buffer = BytesIO(audio_content)
    audio_buffer.name = audio_file.filename
    transcription = openai.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",
        file=audio_buffer
    )
    return transcription

@app.route('/process_audio', methods=['POST'])
async def process_audio():
    request_id = id(request)
    logger.info(f"[Request {request_id}] New audio processing request received")

    try:
        logger.info(f"[Request {request_id}] Parsing request files...")
        files = await request.files
        if 'audio' not in files:
            logger.error(f"[Request {request_id}] Missing audio file in request")
            return jsonify({'error': 'Missing audio file in request'}), 400

        audio_file = files['audio']
        logger.info(f"[Request {request_id}] Sending audio to OpenAI for transcription...")

        transcription_start = asyncio.get_event_loop().time()
        transcription = transcribe(audio_file)
        transcription_duration = asyncio.get_event_loop().time() - transcription_start
        logger.info(f"[Request {request_id}] Transcribed in {transcription_duration:.2f}s: {transcription.text}")

        results = await run_agent(transcription.text, request_id)

        response_data = {
            'success': True,
            'transcription': transcription.text,
            'results': results
        }
        logger.info(f"[Request {request_id}] Request completed successfully")
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"[Request {request_id}] Error processing request: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

async def main():
    await init_agent()

    # NOTE: it seems the `Computer` framework is internally hardcoded to port 8000
    await asyncio.gather(
        computer_server.Server().start_async(),
        app.run_task(host='localhost', port=8001, debug=True)
    )

if __name__ == "__main__":
    asyncio.run(main())