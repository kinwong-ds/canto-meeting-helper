import os
import torch
import torchaudio
from transformers import pipeline

# Load API key from environment variable
HF_API_KEY = os.environ.get("HF_API_KEY")

# Define the pipeline for translation
pipe = pipeline("automatic-speech-recognition",
    model="openai/whisper-large-v3",
    torch_dtype=torch.float16)

# Load the audio file
audio_file = "sample_audio.wav"

# Check if the audio file exists
if not os.path.exists(audio_file):
    raise FileNotFoundError(f"Audio file '{audio_file}' not found.")

# Perform the translation
transcription = pipe(audio_file,  chunk_length_s=30, batch_size=16, return_timestamps=True, generate_kwargs={"language": "yue"})

# Save the translated text to a file
output_file = "translated_text.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(transcription["text"])

print(f"Translation complete. The translated text has been saved to '{output_file}'.")
