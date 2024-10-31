# PROCONF VALERA

## Description

This is source code for my pet project AI Voice Agent. Idea behind this agent is to have local hotword enabled voice assistant that can answer your questions using blazingly fast [OpenAI Realtime API](https://beta.openai.com/docs/api-reference/realtime-api/overview) during our [podcast](https://www.youtube.com/@ProConf/streams) recordings.

By default it does use russian language as a primary language, but it can be easily changed to any other language supported.

## Main Features

- Blazingly fast responses through websockets
- Ability to interrupt current response with new question
- Ability to shutdown agent by voice command

## Requirements

Everything should be pretty straightforward. You will need to download [navec](https://github.com/natasha/navec) embeddings and put them into root folder.

## Technologies

- VAD with [Silero VAD](https://github.com/snakers4/silero-vad)
- STT with [Vosk](https://github.com/alphacep/vosk-api)
- Words embeddings with [Navec](https://github.com/natasha/navec)
- STS with [OpenAI Realtime API](https://beta.openai.com/docs/api-reference/realtime-api/overview)
