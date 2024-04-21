import librosa
import torch
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Path to the directory containing the model files


# Load the model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

# Path to the MP3 file
audio_file_path = "./audio/left-right-up-down.mp3"

# Load audio file using librosa
waveform, sample_rate = librosa.load(audio_file_path, sr=16000, mono=True)  # Resamples to 16000 Hz, converts to mono
waveform = torch.tensor([np.array(waveform)])  # Convert to tensor and add a batch dimension

# Prepare the audio for the model
input_features = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt").input_features

# Generate token ids and decode them to text
predicted_ids = model.generate(input_features)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

print(transcription)
