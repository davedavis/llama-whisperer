# About Llama Whisperer

Google Assistant, Siri, Alexa. Why do they suck so bad? Why have Google,
Apple and Amazon seemingly given up on these platforms that had so much
potential? 

Imagine your voice assistant had the same level of understanding as ChatGPT
along with custom wake words and the ability to fine tune your assistant with
a voice, tone and knowledge that you define. 

Llama Whisperer is a project to create a voice assistant that listens locally,
processes all the voice activity locally and only once it detects your wake 
word and commands does it send the audio transcription to ChatGPT. 

# How Does It Work?
Llama Whisperer uses the local OpenAI whisper model and runs it locally on 
your machine. It uses the local microphone to listen for your wake word and 
then sends the transcript of your audio/question to ChatGPT.

All processing is done locally and no audio is sent to any cloud services. It's
also super fast. No processing sound files one by one and sending them for 
transcription. Transcription is done completely locally. 

Something huge that's been overlooked by many developers is that OpenAi released
the whisper model completely and is free to download and run locally. This means
you can download the model and use it to process speech locally. With a few
lines of python, we can have it listen constantly, transcribe everything we say
and listen for a wake word. Once the wake word is found, it transcribes 
everything after the wake word (locally) and sends it to ChatGPT for processing.

# Dependencies

To use this script, you need to install the following dependencies:
- PortAudio - https://www.portaudio.com/
- PyAudio - https://pypi.org/project/PyAudio/
- OpenAI Whisper Model - This will be downloaded automatically when you run the script
- ffmpg - https://ffmpeg.org/ - sudo apt-get install ffmpg


You can install PortAudio and PyAudio using the following commands:
```

# How do I Use It?
Define your wake word in the .env file.

Then. simply run main.py and it will start listening for your wake word. Once 
it hears your wake word, it will start listening for your question. Once you
stop talking, the script will transcribe your question locally and send it to
ChatGPT for processing.