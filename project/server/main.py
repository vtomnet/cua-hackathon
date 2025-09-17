import asyncio
import sys
import os
import time
import logging
import json
from io import BytesIO
from dotenv import load_dotenv
from openai import OpenAI
from quart import Quart, request, jsonify
from quart_cors import cors
from computer import Computer
from agent import ComputerAgent
from agent.callbacks.base import AsyncCallbackHandler

class CustomLogger(AsyncCallbackHandler):
    def __init__(self):
        pass

    async def on_api_end(self, kwargs, result):
        print("GOT RESPONSE:", result)

load_dotenv()
client = OpenAI()

# TODO this should be a system prompt
USER_PROMPT = """
The user's host operating system is macOS.
- Open applications with Spotlight (cmd+space)
- See open windows and workspaces with Mission Control (ctrl+up arrow)
- They are using Firefox as their browser.
""".strip()

app = Quart(__name__)
app = cors(app, allow_origin="*", allow_methods=["POST", "OPTIONS"], allow_headers=["Content-Type"])

computer = None
agent = None

async def initialize_agent():
    global computer, agent

    if computer is None:
        computer = Computer(use_host_computer_server=True)
        await computer.__aenter__()

    if agent is None:
        # model='omniparser+openai/gpt-4o',
        # model='claude-sonnet-4-20250514'
        agent = ComputerAgent(
            model='omniparser+anthropic/claude-sonnet-4-20250514',
            tools=[computer],
            max_trajectory_budget=5.0,
            use_prompt_caching=True,
            only_n_most_recent_images=3,
            verbosity=logging.INFO,
            callbacks=[
                CustomLogger(),
                # LoggingCallback(
                #     logger=logging.getLogger("cua"),
                #     level=logging.INFO
                # )
            ]
        )

@app.route('/process', methods=['POST'])
async def process_input():
    start_time = asyncio.get_running_loop().time()

    try:
        await initialize_agent()

        files = await request.files
        if 'audio' not in files:
            return jsonify({'error': 'Missing audio file in request'}), 400

        audio_file = files['audio']

        if not audio_file.filename.endswith('.wav'):
            return jsonify({'error': 'Only WAV files are supported'}), 400

        audio_content = audio_file.read()
        audio_buffer = BytesIO(audio_content)
        audio_buffer.name = audio_file.filename
        transcription = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=audio_buffer
        )

        transcript_time = asyncio.get_event_loop().time()
        user_input = transcription.text
        print(f"Got transcription in {(transcript_time - start_time):.2f}s: {user_input}")

        messages = [
            {"role": "user", "content": f"{USER_PROMPT}\n\n{user_input}"}
        ]
        results = []

        async for result in agent.run(messages):
            print("RESULT:", result, file=sys.stderr)
            # for item in result['output']:
            #     if item['type'] == 'message':
            #         message_text = item['content'][0]['text']
            #         results.append(message_text)
            #         print(message_text)

        agent_time = asyncio.get_event_loop().time()
        print(f"Agent finished in {(agent_time - transcript_time):.2f}s, took {len(results)} messages")
        print(f"Total time: {(agent_time - start_time):.2f}s")

        response_data = {
            'success': True,
            'transcription': user_input,
            'results': results
        }
        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    asyncio.run(initialize_agent())
    print("Starting Quart server on http://localhost:8002")
    print("Server ready to accept audio processing requests at /process endpoint")
    sys.stdout.flush()
    app.run(host='localhost', port=8002, debug=True)
