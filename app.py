import os
import logging
import json
import asyncio
import pyaudio
import base64
import colorlog
import queue
import numpy as np
from scipy.signal import resample
from silero_vad import load_silero_vad, VADIterator
from vosk import Model, KaldiRecognizer
from navec import Navec
from websockets.asyncio.client import connect
from dotenv import load_dotenv
from InquirerPy.resolver import prompt
import argparse

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REALTIME_URL = os.getenv("OPENAI_REALTIME_URL")

handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
	'%(log_color)s%(message)s'))

logger = logging.getLogger("PROCONF_VALERA")
logger.setLevel(logging.INFO)
logger.addHandler(handler)

INPUT_SAMPLING_RATE = 48000
VAD_SAMPLING_RATE = 16000
OUTPUT_SAMPLING_RATE = 24000
REALTIME_API_SAMPLING_RATE = 24000

ACTIVATE_PHRASES = ['валера проснись', 'валера привет', 'валера приди']
DEACTIVATE_PHRASES = ['валера отмена', 'валера спасибо']

ACTIVATE_THRESHOLD = 3.65
DEACTIVATE_THRESHOLD = 3.65

navec = Navec.load('navec_hudlit_v1_12B_500K_300d_100q.tar')

ACTIVATE_VECTORS = [np.mean([navec[_] for _ in phrase.split(' ')], axis=0) for phrase in ACTIVATE_PHRASES]
DEACTIVATE_VECTORS = [np.mean([navec[_] for _ in phrase.split(' ')], axis=0) for phrase in DEACTIVATE_PHRASES]

OUTPUT_BUFFER_SIZE = 512
OUTPUT_BUFFER_SIZE_BYTES = OUTPUT_BUFFER_SIZE * 2
OUTPUT_BUFFER_CHANNELS = 1
CALLBACK_SILENCE = OUTPUT_BUFFER_SIZE_BYTES * OUTPUT_BUFFER_CHANNELS * b'\x00'

PROMPT = """
I want you to act as a drunk person.
You will only answer like a very drunk person and nothing else.
Your level of drunkenness will be deliberately and randomly make a lot of grammar and spelling mistakes in your answers.
You will also randomly say something random with the same level of drunkeness I mentioned.
You will randomly answer in aggressive and rude manner.
Use only Russian language for communication.
Use standard accent and familiar dialect for user.
Keep answers short with no more than 50 words.
"""

REALTIME_SESSION_OBJECT = {
    "type": "session.update",
    "session": {
        "modalities": ["text", "audio"],
        "instructions": PROMPT,
        "voice": "ash",
        "input_audio_format": "pcm16",
        "output_audio_format": "pcm16",
        "turn_detection": None,
        "input_audio_transcription": None,
        "max_response_output_tokens": 1024,
        "temperature": 1.0
    }
}

LONG_SILENCE = b'\x00\x00' * INPUT_SAMPLING_RATE * 5

def list_devices():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')

    input_devices = []
    output_devices = []

    for i in range(0, numdevices):
        device_info = p.get_device_info_by_host_api_device_index(0, i)
        if device_info.get('maxInputChannels') > 0:
            input_devices.append((i, device_info.get('name')))
        if device_info.get('maxOutputChannels') > 0:
            output_devices.append((i, device_info.get('name')))

    p.terminate()
    return input_devices, output_devices

def select_device(devices, device_type):
    choices = [{"name": f"{index}: {name}", "value": index} for index, name in devices]
    questions = [
        {
            "type": "list",
            "message": f"Select {device_type} device:",
            "choices": choices,
        }
    ]
    selected_device = prompt(questions)[0]
    return selected_device

class ProconfAgent:
    def __init__(self, loop, input_device_index, output_device_index):
        self._input_buffer = asyncio.Queue()
        self._output_buffer = queue.Queue()
        self._websocket_buffer = asyncio.Queue()

        self._leftover_buffer = None

        self._closed = True
        self._loop = loop
        self._input_device_index = input_device_index
        self._output_device_index = output_device_index
        self._vad_iterator = VADIterator(load_silero_vad(), sampling_rate=VAD_SAMPLING_RATE)
        self._speaking = False
        self._active = False
        self._websocket_task = None
        self._websocket_response_in_progress = False
        self._rec = KaldiRecognizer(Model(model_name="vosk-model-small-ru-0.22"), INPUT_SAMPLING_RATE)

    async def __aenter__(self):
        await self.__open()
        return self

    async def __aexit__(self, type, value, traceback):
        await self.__close()

    def send_chunk(self, status, chunk):
        self._loop.call_soon_threadsafe(self._input_buffer.put_nowait, (status, chunk))
        return True

    def __output_callback(self, in_data, frame_count, time_info, status):
        if self._output_buffer.empty():
            return (CALLBACK_SILENCE, pyaudio.paContinue)

        chunk = self._output_buffer.get()

        if len(chunk) < OUTPUT_BUFFER_SIZE_BYTES:
            logger.error("SIZE IS INCORRECT")
            return (CALLBACK_SILENCE, pyaudio.paContinue)

        return (chunk, pyaudio.paContinue)

    def __callback(self, in_data, frame_count, time_info, status):
        audio_float32 = np.frombuffer(in_data, dtype=np.float32)

        number_of_samples = round(len(audio_float32) * float(VAD_SAMPLING_RATE) / INPUT_SAMPLING_RATE)
        resampled_audio_float32 = resample(audio_float32, number_of_samples)

        speech_dict = self._vad_iterator(resampled_audio_float32, return_seconds=True)

        if speech_dict and ('start' in speech_dict):
            logger.info("START SPEAKING")
            self._speaking = True
            self.send_chunk('start', audio_float32.tobytes())
            return None, pyaudio.paContinue

        if self._speaking:
            self.send_chunk('chunk', audio_float32.tobytes())

        if speech_dict and ('end' in speech_dict):
            logger.info("END SPEAKING")
            self._speaking = False
            self.send_chunk('end', LONG_SILENCE)
            return None, pyaudio.paContinue

        return None, pyaudio.paContinue

    async def __open(self):
        self._pyaudio_obj = pyaudio.PyAudio()
        self._input = self._pyaudio_obj.open(
            format=pyaudio.paFloat32,
            input=True,
            channels=1,
            rate=INPUT_SAMPLING_RATE,
            input_device_index=self._input_device_index,
            frames_per_buffer=512 * 3,
            stream_callback=self.__callback,
        )
        self._output = self._pyaudio_obj.open(
            format=pyaudio.paInt16,
            output=True,
            channels=1,
            rate=OUTPUT_SAMPLING_RATE,
            output_device_index=self._output_device_index,
            frames_per_buffer=OUTPUT_BUFFER_SIZE,
            stream_callback=self.__output_callback,
            start=False,
        )
        self._closed = False

    async def __close(self):
        self._vad_iterator.reset_states()
        self._input.stop_stream()
        self._output.stop_stream()
        self._input.close()
        self._output.close()
        self._closed = True
        self._pyaudio_obj.terminate()
        del(self._input_buffer)
        del(self._output_buffer)
        del(self._websocket_buffer)

    async def run(self):
        await self.listen()

    def clear_output_buffer(self):
        while not self._output_buffer.empty():
            self._output_buffer.get_nowait()
        self._leftover_buffer = None

    def clear_websocket_buffer(self):
        while not self._websocket_buffer.empty():
            self._websocket_buffer.get_nowait()

    async def read_websocket(self, websocket):
        async for msg in websocket:
            message = json.loads(msg)

            logger.debug(message["type"])

            match message:
                case {"type": "response.audio.delta"}:
                    audio = np.frombuffer(base64.b64decode(message["delta"]), dtype=np.int16).tobytes()

                    if self._leftover_buffer is not None:
                        audio = self._leftover_buffer + audio
                        self._leftover_buffer = None

                    for i in range(0, len(audio), OUTPUT_BUFFER_SIZE_BYTES):
                        chunk = audio[i:i + OUTPUT_BUFFER_SIZE_BYTES]

                        if len(chunk) < OUTPUT_BUFFER_SIZE_BYTES:
                            self._leftover_buffer = chunk
                            break

                        self._loop.call_soon_threadsafe(self._output_buffer.put_nowait, (chunk))

                case { "type": "response.created" }:
                    self._websocket_response_in_progress = True

                case {"type": "response.done", "response": response}:
                    self._websocket_response_in_progress = False

                    if response["status"] == "failed":
                        logger.error("FAILED RESPONSE {response}".format(response=json.dumps(response["status_details"])))

    async def write_websocket(self, websocket):
        while True:
            message = await self._websocket_buffer.get()
            await websocket.send(message)

    async def connect_to_websocket(self):
        async for websocket in connect(uri=REALTIME_URL, additional_headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "OpenAI-Beta": "realtime=v1"}):
            try:
                self._websocket_read_task = self._loop.create_task(self.read_websocket(websocket))
                self._websocket_write_task = self._loop.create_task(self.write_websocket(websocket))

                await asyncio.gather(self._websocket_read_task, self._websocket_write_task)
            except asyncio.CancelledError:
                logger.debug("CANCELLED")
                self.clear_websocket_buffer()
                self.clear_output_buffer()
                await websocket.close()
                self._websocket_task = None
                raise

    def send_to_websocket(self, message):
        self._loop.call_soon_threadsafe(self._websocket_buffer.put_nowait, message)

    async def listen(self):
        async for action, chunk in self.process():
            if action == "activate":
                logger.info("ACTIVATE")
                self._active = True
                if self._websocket_task is None:
                    self._websocket_task = self._loop.create_task(self.connect_to_websocket())
                    self.send_to_websocket(json.dumps(REALTIME_SESSION_OBJECT))
                    self.send_to_websocket('{"type": "input_audio_buffer.clear"}')
                    self._output.start_stream()

            if action == "deactivate":
                logger.info("DEACTIVATE")
                self._active = False

                self.clear_websocket_buffer()
                self.clear_output_buffer()
                self._output.stop_stream()

                if self._websocket_task is not None:
                    self._websocket_task.cancel()

            if action == "interrupt":
                logger.info("INTERRUPT")
                self.clear_websocket_buffer()
                self.clear_output_buffer()
                self.send_to_websocket(json.dumps({ "type": "response.cancel" }))

            if action == "start":
                self.clear_websocket_buffer()
                self.send_to_websocket(json.dumps({ "type": "input_audio_buffer.clear" }))

            if action == "start" or action == "chunk" or action == "end":
                audio = np.frombuffer(chunk, dtype=np.float32)

                number_of_samples = round(len(audio) * float(REALTIME_API_SAMPLING_RATE) / INPUT_SAMPLING_RATE)
                resampled_chunk = resample(audio, number_of_samples)
                resampled_chunk_int16 = (resampled_chunk * 32768).astype(np.int16).tobytes()

                base64_chunk = base64.b64encode(resampled_chunk_int16).decode()

                message = { "type": "input_audio_buffer.append", "audio": base64_chunk }
                self.send_to_websocket(json.dumps(message))

            if action == "end":
                self.send_to_websocket(json.dumps({ "type": "input_audio_buffer.commit" }))
                self.send_to_websocket(json.dumps({ "type": "response.create" }))

    async def process(self):
        async for status, chunk in self.microphone():
            chunk_int16 = (np.frombuffer(chunk, dtype=np.float32) * 32768).astype(np.int16).tobytes()

            if self._rec.AcceptWaveform(chunk_int16):
                result = json.loads(self._rec.Result())["text"].split(' ')[-2:]
                vectorization_failed = False

                logger.info(result)

                try:
                    result_vector = np.mean([navec[_] for _ in result], axis=0)
                except KeyError:
                    vectorization_failed = True
                    logger.error("UNKNOWN WORD IN RESULT %s".format(str(result)))


                if not vectorization_failed:
                    if any([np.linalg.norm(vector - result_vector) < ACTIVATE_THRESHOLD for vector in ACTIVATE_VECTORS]):
                        yield "activate", None
                        continue

                    if any([np.linalg.norm(vector - result_vector) < DEACTIVATE_THRESHOLD for vector in DEACTIVATE_VECTORS]):
                        yield "deactivate", None
                        continue

            if status == "start" and (self._websocket_response_in_progress or not self._output_buffer.empty()):
                yield "interrupt", None

            if self._active:
                yield status, chunk

    async def microphone(self):
        while not self._closed:
            try:
                status, chunk = await self._input_buffer.get()
                yield status, chunk
            except asyncio.QueueEmpty:
                if self._loop is not None:
                    self._loop.stop()
                break

async def run(input_device_index, output_device_index):
    loop = asyncio.get_running_loop()

    async with ProconfAgent(loop, input_device_index, output_device_index) as valera:
        await valera.run()

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="ProconfAgent CLI")
        parser.add_argument("--input_device_index", type=int, help="Index of the input device")
        parser.add_argument("--output_device_index", type=int, help="Index of the output device")
        args = parser.parse_args()

        input_device_index = args.input_device_index
        output_device_index = args.output_device_index

        if input_device_index is None or output_device_index is None:
            input_devices, output_devices = list_devices()

            if input_device_index is None:
                input_device_index = select_device(input_devices, "input")

            if output_device_index is None:
                output_device_index = select_device(output_devices, "output")

        asyncio.run(run(input_device_index, output_device_index))
    except KeyboardInterrupt:
        print("Terminating...")
