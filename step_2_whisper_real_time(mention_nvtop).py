import asyncio
import pyaudio
import wave
import whisper

# Parameters for recording audio
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
# 1024 for 2 seconds.  adjust for longer or shorter chunks
CHUNK = 2048

# Initialize the audio interface
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

# Load the Whisper model. Details of which one to choose can be found here:
# https://github.com/openai/whisper#available-models-and-languages
# Note, whisper library/package will automatically download the model when it
# is first initialized below, so no need to download anything.
model = whisper.load_model("medium")


async def record_audio(filename):
    """
    An asynchronous generator that records audio continuously and yields
    filenames of audio chunks. Each chunk is approximately 4 seconds long.
    """
    while True:
        # Record a chunk of audio
        frames = []
        for _ in range(0, int(RATE / CHUNK * 4)):  # record for 4 seconds
            data = stream.read(CHUNK)
            frames.append(data)

        # Write the chunk to a file
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))

        # Yield the filename
        yield filename

        # Yield control to the event loop, allowing other asynchronous
        # operations to proceed
        await asyncio.sleep(0)


async def transcribe_audio():
    """
    Continuously transcribes audio from the microphone. Audio is recorded
    in chunks (approximately 4 seconds each), and each chunk is transcribed
    separately.
    """
    # Create the asynchronous generator
    audio_generator = record_audio("chunk.wav")

    # Loop over chunks of audio
    async for filename in audio_generator:
        # Load audio from the file
        audio = whisper.load_audio(filename)

        # Pad or trim the audio to fit the model's requirements
        audio = whisper.pad_or_trim(audio)

        # Convert the audio to a log-Mel spectrogram
        mel = whisper.log_mel_spectrogram(audio).to(model.device)

        # Transcribe the audio
        options = whisper.DecodingOptions()
        result = whisper.decode(model, mel, options)

        # Print the transcription
        print("Transcription:", result.text)


def main():
    """
    The main function of the script. Starts the transcription process.
    """
    asyncio.run(transcribe_audio())


if __name__ == "__main__":
    main()
