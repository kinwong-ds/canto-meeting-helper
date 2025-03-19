import streamlit as st
from streamlit_scroll_navigation import scroll_navbar
import os
import numpy as np
import librosa
import csv
import wave
import soundfile
import requests
import pandas as pd
from datetime import timedelta
from pydub import AudioSegment
import speech_recognition as sr
from transformers import pipeline
from openai import AzureOpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import torch
import google.generativeai as genai
import pickle
from scipy.signal import medfilt

# Load environment variables from .env file
load_dotenv()

# Get the Hugging Face API key from the environment variables
HF_API_KEY = os.environ.get("HF_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Determine the best available device (CUDA if available, otherwise MPS, then CPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# device = torch.device("cpu")

print(f"Using device: {device}")
torch.set_default_device(device)

# --- Functions from Notebook ---
def convert_audio_to_wav(input_file):
    """Convert .m4a file to .wav format."""
    # Load the .m4a file
    audio = AudioSegment.from_file(input_file, format="m4a")
    output_file = os.path.join("converted_audio", os.path.basename(input_file).replace(".m4a", ".wav"))
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    audio.export(output_file, format="wav")
    return output_file

def split_audio_by_silence(audio_path, output_dir, chunk_duration=300, min_silence_duration=0.5,
                           silence_threshold=-40, padding=0.2):
    """
    Split an audio file into chunks based on silence detection to avoid splitting in the middle of sentences.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # Load the audio file with pydub
    audio = AudioSegment.from_file(audio_path)
    # Convert to mono if stereo for easier processing
    if audio.channels > 1:
        audio = audio.set_channels(1)
    # Get the duration in milliseconds
    total_duration_ms = len(audio)
    # Convert chunk duration to milliseconds
    chunk_duration_ms = chunk_duration * 1000
    # Load the audio file with librosa for silence detection
    y, sr = librosa.load(audio_path, sr=None)
    # Compute the RMS energy
    rms = librosa.feature.rms(y=y, frame_length=int(sr * 0.025), hop_length=int(sr * 0.01))[0]
    # Convert to dB
    rms_db = 20 * np.log10(rms + 1e-10)
    # Apply median filtering to smooth the energy curve
    rms_db_smoothed = medfilt(rms_db, kernel_size=11)
    # Find silence regions (below threshold)
    silence_mask = rms_db_smoothed < silence_threshold
    # Convert to time (seconds)
    times = librosa.times_like(rms_db_smoothed, sr=sr, hop_length=int(sr * 0.01))
    # Find continuous silence regions
    silence_regions = []
    in_silence = False
    start_time = 0
    for i, is_silent in enumerate(silence_mask):
        if is_silent and not in_silence:
            # Start of a silence region
            start_time = times[i]
            in_silence = True
        elif not is_silent and in_silence:
            # End of a silence region
            end_time = times[i]
            silence_duration = end_time - start_time
            if silence_duration >= min_silence_duration:
                silence_regions.append((start_time, end_time))
            in_silence = False
    # Add the last silence region if we're still in silence at the end
    if in_silence:
        end_time = times[-1]
        silence_duration = end_time - start_time
        if silence_duration >= min_silence_duration:
            silence_regions.append((start_time, end_time))
    # Sort silence regions by their start time
    silence_regions.sort(key=lambda x: x[0])
    # Determine the split points
    split_points = []
    current_chunk_end_ms = chunk_duration_ms
    # Find the closest silence region to each target chunk end
    for i in range(len(silence_regions)):
        silence_start_ms = silence_regions[i][0] * 1000
        silence_end_ms = silence_regions[i][1] * 1000
        # If this silence region is close to our target chunk end
        if abs(silence_start_ms - current_chunk_end_ms) < chunk_duration_ms * 0.2:
            split_points.append(silence_start_ms)
            current_chunk_end_ms += chunk_duration_ms
        # If we've already passed our target chunk end, use this silence region
        elif silence_start_ms > current_chunk_end_ms:
            split_points.append(silence_start_ms)
            current_chunk_end_ms = silence_start_ms + chunk_duration_ms
    # Add the start of the audio as the first split point
    split_points = [0] + split_points
    # Add the end of the audio as the last split point
    if split_points[-1] < total_duration_ms:
        split_points.append(total_duration_ms)
    # Generate chunks
    chunk_paths = []
    for i in tqdm(range(len(split_points) - 1), desc="Splitting audio into chunks"):
        # Calculate start and end times with padding
        start_ms = max(0, split_points[i] - padding * 1000)
        end_ms = min(total_duration_ms, split_points[i + 1] + padding * 1000)
        # Extract the chunk
        chunk = audio[start_ms:end_ms]
        # Generate the output filename
        output_filename = f"chunk_{i+1:03d}.wav"
        output_path = os.path.join(output_dir, output_filename)
        # Export the chunk
        chunk.export(output_path, format="wav")
        chunk_paths.append(output_path)
        # Print progress
        print(f"Chunk {i+1}/{len(split_points)-1} - Duration: {(end_ms-start_ms)/1000:.2f}s")
    return chunk_paths

def get_all_file_paths(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):  # Make sure to only include .wav files
                file_paths.append(os.path.join(root, file))
    # Sort the file paths based on the numeric part of the filename
    file_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))
    return file_paths

def transcribe_with_gemini(audio_chunk_paths, api_key):
    try:
        print("Transcribing with Gemini Pro 1.5...")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-pro')
        # Create prompt for diarization and Cantonese transcription
        prompt = """
        Transcribe this Cantonese audio in Hong Kong colloquial speech, removing filler words and repeated phrases. Extract only spoken content and ignore background noise.
        """
        transcriptions = []
        progress_bar = st.progress(0)
        for i, chunk_path in enumerate(tqdm(audio_chunk_paths, desc="Transcribing chunks")):
            # Upload audio file with specified MIME type
            f = genai.upload_file(chunk_path, mime_type='audio/wav')
            # Generate using multimodal capabilities
            response = model.generate_content([prompt, f])
            transcriptions.append(response.text)
            progress_bar.progress((i + 1) / len(audio_chunk_paths))
        return "".join(transcriptions)
    except Exception as e:
        print(f"Error with Gemini transcription: {str(e)}")
        return None

def gemini_prompt_call(message, api_key, prompt, temperature=0.3, top_p=0.8):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-pro')

        prompt_parts = [prompt+'\n', message]

        generation_config = genai.GenerationConfig(
                            temperature=temperature,
                            top_p=top_p,
                            )  

        response = model.generate_content(prompt_parts, generation_config=generation_config)
        return response.text

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def refine_cantonese(message, api_key):
    prompt = """
            You are an expert in Cantonese. Please perform the following two steps and return the revised text.
            1. Correct potential inaccuracies in the Cantonese text in Hong Kong style. For example, in Hong Kong, text such as '噉', should be written as '咁'.
            2. Refine punctuation to the text.
            3. Remove repeated and unnecessary words.  For example, '係嘅係嘅係嘅' or '啱嘅啱嘅啱嘅' can be replaced with '係'.
            """
    return gemini_prompt_call(message, api_key, prompt)

def speech_to_written(message, api_key):
    prompt = """
            Convert the following spoken Cantonese text into standard written Chinese in traditional characters (zh-Hant). Maintain grammatical correctness and coherence while ensuring the meaning is preserved. Use formal written Chinese (書面語) rather than colloquial Cantonese (口語).
            """
    return gemini_prompt_call(message, api_key, prompt)

def refine_writte_text(message, api_key):
    prompt = """
            Please refine the following Chinese text by making the sentence structure smoother and more coherent. 
            Reduce excessive punctuation, clarify ambiguous parts, and ensure natural readability while preserving the original meaning.
            Output in traditional Chinese characters (zh-Hant).
            """
    return gemini_prompt_call(message, api_key, prompt, temperature=1)

def summarize_refined_written(message, api_key, sum_lvl, bullet_points=False):
    if bullet_points:
        prompt = f"""Summarize the given Chinese text and output in Traditional Chinese (zh) markdown format with bullet points and sectioned format. Use detail level {sum_lvl}, where 1 is the most concise and 5 is the most detailed. Retain key information and context while ensuring clarity, coherence, and fluency."""
    else:
        prompt = f"""Summarize the given Chinese text and output in Traditional Chinese (zh) markdown format at summarization level {sum_lvl}, where 1 is the most concise and 5 is the most detailed. Retain key information and context while ensuring clarity, coherence, and fluency."""
    return gemini_prompt_call(message, api_key, prompt)

# --- Streamlit App ---
def main():
    st.title("Cantonese Speech-to-Text and Summarization")

    with st.sidebar:
        scroll_navbar(anchor_ids=["upload_audio"], key="upload_audio_navbar")
        scroll_navbar(anchor_ids=["transcription"], key="transcription_navbar")
        scroll_navbar(anchor_ids=["summarization"], key="summarization_navbar")
        scroll_navbar(anchor_ids=["save_summary"], key="save_summary_navbar")

    st.subheader("Upload Audio", anchor="upload_audio")
    uploaded_file = st.file_uploader("Select raw .m4a file for upload", type=["m4a"])
    if uploaded_file is not None:
        # Save the uploaded file
        input_m4a_audio = os.path.join("input_audio", uploaded_file.name)
        os.makedirs("input_audio", exist_ok=True)
        with open(input_m4a_audio, "wb") as f:
            f.write(uploaded_file.read())
        st.success("File uploaded successfully!")

    st.subheader("Transcription Result", anchor="transcription")
    transcription_result = st.empty()  # Placeholder for transcription result
    if st.button("Transcribe"):
        if uploaded_file is not None:
            # --- Audio Processing ---
            try:
                input_m4a_audio = os.path.join("input_audio", uploaded_file.name)
                raw_wav_file = convert_audio_to_wav(input_m4a_audio)
                file_name = uploaded_file.name.split('.')[0]
                chunk_paths = split_audio_by_silence(raw_wav_file, f"chunked_audio/{file_name}", chunk_duration=60)
                # --- Transcription ---
                audio_chunk_paths = get_all_file_paths(f'chunked_audio/{file_name}')
                gemini_output = transcribe_with_gemini(audio_chunk_paths, GEMINI_API_KEY)
                # --- Refinement ---
                refined_spoken_text = refine_cantonese(gemini_output, GEMINI_API_KEY)
                raw_written = speech_to_written(refined_spoken_text, GEMINI_API_KEY)
                refined_written = refine_writte_text(raw_written, GEMINI_API_KEY)
                transcription_result.write(refined_written)
            except Exception as e:
                st.error(f"An error occurred during transcription: {e}")
        else:
            st.warning("Please upload an audio file first.")

    st.subheader("Summarization", anchor="summarization")
    summary_detail_level = st.slider("Select summary detail level (1-5)", 1, 5, 3)
    bullet_point_style = st.checkbox("Output in bullet point style")
    summary_result = st.empty()  # Placeholder for summary result
    if st.button("Summarize"):
        if uploaded_file is not None:
            try:
                summarized_text = summarize_refined_written(refined_written, GEMINI_API_KEY, sum_lvl=summary_detail_level, bullet_points=bullet_point_style)
                summary_result.write(summarized_text)
            except Exception as e:
                st.error(f"An error occurred during summarization: {e}")
        else:
            st.warning("Please transcribe an audio file first.")

    st.subheader("Save Summary", anchor="save_summary")
    if st.button("Save Summary"):
        if 'summarized_text' in locals():
            try:
                file_name = uploaded_file.name.split('.')[0]
                save_summary(summarized_text, filename=f"{file_name}_summary.md")
            except Exception as e:
                st.error(f"An error occurred during saving: {e}")
        else:
            st.warning("Please generate a summary first.")

if __name__ == "__main__":
    import librosa
    from scipy.signal import medfilt
    main()
