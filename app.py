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

# === Streamlit App ===
def main():
    st.title("🎙️ Voice Cloning App")
    st.write("Clone voices and compare similarity with the original")

    # Initialize TTS model
    if 'tts' not in st.session_state:
        with st.spinner("Loading TTS model..."):
            st.session_state.tts = TTS(
                model_name="tts_models/multilingual/multi-dataset/your_tts",
                progress_bar=False,
                gpu=torch.cuda.is_available()
            )

    # Input method selection
    input_method = st.radio(
        "How do you want to provide the voice/text data?",
        options=[
            "Upload audio and text manually",
            "Enter local paths",
            "Use Google Drive link",
            "Upload existing CSV file"
        ]
    )

    wav_file = None
    input_text = None
    csv_data = None

    if input_method == "Upload audio and text manually":
        audio_file = st.file_uploader("Upload your audio (MP3) file", type=["mp3"])
        text_file = st.file_uploader("Upload your text file", type=["txt"])
        
        if audio_file and text_file:
            wav_file = "input.wav"
            with open("temp.mp3", "wb") as f:
                f.write(audio_file.getbuffer())
            convert_mp3_to_wav("temp.mp3", wav_file)
            
            input_text = text_file.read().decode("utf-8")

    elif input_method == "Enter local paths":
        mp3_path = st.text_input("Enter path to your MP3 file")
        text_path = st.text_input("Enter path to your text file")
        
        if mp3_path and text_path:
            wav_file = mp3_path.replace(".mp3", ".wav")
            convert_mp3_to_wav(mp3_path, wav_file)
            
            with open(text_path, 'r') as file:
                input_text = file.read()

    elif input_method == "Use Google Drive link":
        gdrive_url = st.text_input("Enter the Google Drive MP3 link")
        input_text = st.text_area("Enter the text to be spoken using cloned voice")
        
        if gdrive_url and input_text:
            mp3_file = "input.mp3"
            wav_file = "input.wav"
            try:
                file_id = gdrive_url.split("/d/")[1].split("/")[0]
                download_url = f"https://drive.google.com/uc?id={file_id}"
                gdown.download(download_url, mp3_file, quiet=False)
                convert_mp3_to_wav(mp3_file, wav_file)
            except Exception as e:
                st.error(f"Error downloading from Google Drive: {e}")

    elif input_method == "Upload existing CSV file":
        csv_file = st.file_uploader("Upload your voice_dataset.csv", type=["csv"])
        if csv_file:
            csv_data = pd.read_csv(csv_file)
            st.write("Uploaded CSV data:")
            st.dataframe(csv_data)

    # Process cloning if we have the required inputs
    if csv_data is not None:
        st.success("✅ You uploaded an existing CSV, skipping voice cloning.")
    elif wav_file and input_text:
        try:
            language = detect(input_text)
            st.write(f"Detected language: {language}")

            if st.button("Start Voice Cloning"):
                best_similarity = 0
                best_output = ""
                results = []

                st.write("🔁 Running 5 cloning attempts for best match...")
                progress_bar = st.progress(0)
                
                for i in range(5):
                    with st.spinner(f"Running attempt {i+1}/5..."):
                        sim, out_file = clone_and_compare(
                            st.session_state.tts, 
                            wav_file, 
                            input_text, 
                            language, 
                            f"clone_try_{i}.wav"
                        )
                    results.append({"Attempt": i + 1, "Similarity": sim})
                    progress_bar.progress((i+1)/5)
                    st.write(f"Attempt {i+1}: Similarity = {sim*100:.2f}%")

                    if sim > best_similarity:
                        best_similarity = sim
                        best_output = out_file

                # Standardize & Save Final Audio
                standardize_audio_format(best_output, "final_cloned_voice.wav")
                st.success(f"✅ Best voice with similarity {best_similarity*100:.2f}%")

                # Save CSV
                df = pd.DataFrame(results)
                df.to_csv("voice_dataset.csv", index=False)

                # Plot
                fig, ax = plt.subplots()
                ax.plot(df['Attempt'], df['Similarity'] * 100, marker='o')
                ax.set_title("Voice Similarity Over Attempts")
                ax.set_xlabel("Attempt")
                ax.set_ylabel("Similarity (%)")
                ax.set_ylim(0, 100)
                ax.grid(True)
                st.pyplot(fig)

                # Download options
                st.subheader("📥 Download Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    with open("voice_dataset.csv", "rb") as f:
                        st.download_button(
                            "Download CSV",
                            f,
                            file_name="voice_dataset.csv",
                            mime="text/csv"
                        )
                
                with col2:
                    with open("final_cloned_voice.wav", "rb") as f:
                        st.download_button(
                            "Download Audio",
                            f,
                            file_name="final_cloned_voice.wav",
                            mime="audio/wav"
                        )
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
