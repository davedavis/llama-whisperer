from optimum.bettertransformer import BetterTransformer
from transformers import AutoProcessor, BarkModel
from dotenv import load_dotenv
import pyaudio
import torch

load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"

def play_audio(audio_array, sample_rate=22050):
    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open a streaming stream
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=sample_rate,
                    output=True)

    # Play the audio stream
    stream.write(audio_array.astype('float32').tobytes())

    # Close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

# processor = AutoProcessor.from_pretrained("suno/bark")
processor = AutoProcessor.from_pretrained("suno/bark")
# model = BarkModel.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark", torch_dtype=torch.float16).to(device)

# convert to bettertransformer
model = BetterTransformer.transform(model, keep_original_model=False)

# enable CPU offload
model.enable_cpu_offload()

voice_preset = "v2/en_speaker_9"


inputs = processor("Hello Dave, Hope you're well. Seems this is working, although much slower than anticipated [laughs]", voice_preset=voice_preset)
inputs = {k: v.to(device) for k, v in inputs.items()}

audio_array = model.generate(**inputs)
audio_array = audio_array.cpu().numpy().squeeze()

inputs2 = processor("Dave [laughs], this might speed things up a little. If I'm not mistaken. [excited]Hopefully we can do some much better inference at real time speeds. [laugh]Peace out", voice_preset=voice_preset)
inputs2 = {k: v.to(device) for k, v in inputs2.items()}

audio_array2 = model.generate(**inputs2)
audio_array2 = audio_array2.cpu().numpy().squeeze()

# Play the audio
play_audio(audio_array)
play_audio(audio_array2)