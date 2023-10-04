import whisper
import pyaudio
import wave

# from dotenv import load_dotenv
#
# load_dotenv()
# openai.api_key = os.getenv('OPENAI_API_KEY')

# Setup model to load into vram.
model = whisper.load_model("medium")


# Record audio
def record_audio(filename, duration=20):
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024

    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    print("Recording...")

    frames = []

    for _ in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Finished recording")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))


# Transcribe audio
def transcribe_audio(filename):
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(filename)

    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    # ToDo: Move this globally.
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)

    # print the recognized text
    return result.text


# Main function
def main():
    audio_filename = "recorded_audio.wav"
    record_audio(audio_filename)
    transcription = transcribe_audio(audio_filename)
    print("Transcription:", transcription)


if __name__ == "__main__":
    main()
