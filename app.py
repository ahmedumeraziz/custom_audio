import os
import numpy as np
import librosa
from pydub import AudioSegment
import soundfile as sf
import gdown
from TTS.api import TTS
from langdetect import detect
from scipy.spatial.distance import cosine
import torch
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from io import BytesIO

# === Utility Functions ===

def download_mp3_from_gdrive(gdrive_url, output_path="downloaded_voice.mp3"):
    file_id = gdrive_url.split("/d/")[1].split("/")[0]
    download_url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(download_url, output_path, quiet=False)

def convert_mp3_to_wav(mp3_file, wav_file):
    audio = AudioSegment.from_file(mp3_file, format="mp3")
    audio.export(wav_file, format="wav")

def extract_mfcc(wav_file):
    y, sr = librosa.load(wav_file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1)

def clone_and_compare(tts, ref_wav, text, language, output_wav="cloned.wav"):
    tts.tts_to_file(text=text, speaker_wav=ref_wav, language=language, file_path=output_wav)
    orig = extract_mfcc(ref_wav)
    clone = extract_mfcc(output_wav)
    similarity = 1 - cosine(orig, clone)
    return similarity, output_wav

def standardize_audio_format(input_file, output_file, sample_rate=22050):
    y, sr = librosa.load(input_file, sr=sample_rate)
    sf.write(output_file, y, sample_rate)

# === Load Model ===
tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False, gpu=torch.cuda.is_available())

# === Streamlit UI ===
st.title("Voice Cloning and Comparison")

input_method = st.selectbox(
    "How do you want to provide the voice/text data?",
    ["Upload audio and text manually", "Enter local paths", "Use Google Drive link", "Upload existing CSV file"]
)

wav_file = None
input_text = ""
csv_data = None

if input_method == "Upload audio and text manually":
    uploaded_audio = st.file_uploader("Upload your audio (MP3)", type=["mp3"])
    if uploaded_audio:
        mp3_file = uploaded_audio.name
        wav_file = mp3_file.replace(".mp3", ".wav")
        with open(mp3_file, "wb") as f:
            f.write(uploaded_audio.getbuffer())
        convert_mp3_to_wav(mp3_file, wav_file)

    uploaded_text = st.file_uploader("Upload a text file", type=["txt"])
    if uploaded_text:
        input_text = uploaded_text.getvalue().decode("utf-8")

elif input_method == "Enter local paths":
    mp3_file = st.text_input("Enter local path to your MP3 file")
    if mp3_file:
        wav_file = mp3_file.replace(".mp3", ".wav")
        convert_mp3_to_wav(mp3_file, wav_file)
    
    text_file = st.text_input("Enter local path to your text file")
    if text_file:
        with open(text_file, "r") as file:
            input_text = file.read()

elif input_method == "Use Google Drive link":
    gdrive_url = st.text_input("Enter the Google Drive MP3 link")
    if gdrive_url:
        mp3_file = "input.mp3"
        wav_file = "input.wav"
        download_mp3_from_gdrive(gdrive_url, mp3_file)
        convert_mp3_to_wav(mp3_file, wav_file)
        input_text = st.text_area("Enter the text to be spoken using cloned voice")

elif input_method == "Upload existing CSV file":
    uploaded_csv = st.file_uploader("Upload your voice_dataset.csv file", type=["csv"])
    if uploaded_csv:
        csv_data = pd.read_csv(uploaded_csv)
        st.write("CSV uploaded. Here is the data:", csv_data)

if csv_data is None and wav_file and input_text:
    language = detect(input_text)
    st.write(f"Detected language: {language}")

    best_similarity = 0
    best_output = ""
    results = []

    st.write("\n🔁 Running 5 cloning attempts for best match...\n")
    for i in range(5):
        sim, out_file = clone_and_compare(tts, wav_file, input_text, language, f"clone_try_{i}.wav")
        results.append({"Attempt": i + 1, "Similarity": sim})
        st.write(f"Attempt {i+1}: Similarity = {sim*100:.2f}%")

        if sim > best_similarity:
            best_similarity = sim
            best_output = out_file

    # Standardize & Save Final Audio
    standardize_audio_format(best_output, "final_cloned_voice.wav")
    st.write(f"\n✅ Best voice saved as 'final_cloned_voice.wav' with similarity {best_similarity*100:.2f}%")

    # Save CSV
    df = pd.DataFrame(results)
    df.to_csv("voice_dataset.csv", index=False)
    st.write("📄 Similarity results saved as 'voice_dataset.csv'")

    # Plot
    plt.plot(df['Attempt'], df['Similarity'] * 100, marker='o')
    plt.title("Voice Similarity Over Attempts")
    plt.xlabel("Attempt")
    plt.ylabel("Similarity (%)")
    plt.ylim(0, 100)
    plt.grid(True)
    st.pyplot(plt)

    # === Final Download Menu ===
    download_choice = st.radio("Choose a file to download:", ["Similarity CSV", "Best Cloned Audio", "Both"])
    
    if download_choice == "Similarity CSV":
        with open("voice_dataset.csv", "rb") as f:
            st.download_button("Download CSV", f, file_name="voice_dataset.csv")
    elif download_choice == "Best Cloned Audio":
        with open("final_cloned_voice.wav", "rb") as f:
            st.download_button("Download Cloned Audio", f, file_name="final_cloned_voice.wav")
    elif download_choice == "Both":
        with open("voice_dataset.csv", "rb") as f:
            st.download_button("Download CSV", f, file_name="voice_dataset.csv")
        with open("final_cloned_voice.wav", "rb") as f:
            st.download_button("Download Cloned Audio", f, file_name="final_cloned_voice.wav")
