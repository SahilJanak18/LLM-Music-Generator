from audiocraft.models import MusicGen
import streamlit as st
import torch
import torchaudio
import os
import numpy as np
import base64
import json

@st.cache_resource
def load_model(model_name='facebook/musicgen-small'):
    model = MusicGen.get_pretrained(model_name)
    return model

def generate_music_tensors(description, duration: int, model_name='facebook/musicgen-small'):
    print("Description: ", description)
    print("Duration: ", duration)
    model = load_model(model_name)

    model.set_generation_params(
        use_sampling=True,
        top_k=250,
        duration=duration
    )

    output = model.generate(
        descriptions=[description],
        progress=True,
        return_tokens=True
    )

    return output[0]

def apply_audio_effects(samples: torch.Tensor, effect: str):
    sample_rate = 32000
    if effect == "Reverb":
        samples, _ = torchaudio.sox_effects.apply_effects_tensor(samples, sample_rate, [["reverb", "50"]])
    elif effect == "Echo":
        samples, _ = torchaudio.sox_effects.apply_effects_tensor(samples, sample_rate, [["echo", "0.8", "0.88", "60", "0.4"]])
    return samples

def save_audio(samples: torch.Tensor, save_path: str = "audio_output/"):
    sample_rate = 32000
    os.makedirs(save_path, exist_ok=True)
    assert samples.dim() == 2 or samples.dim() == 3

    samples = samples.detach().cpu()
    if samples.dim() == 2:
        samples = samples[None, ...]

    for idx, audio in enumerate(samples):
        audio_path = os.path.join(save_path, f"audio_{idx}.wav")
        torchaudio.save(audio_path, audio, sample_rate)
    
    return audio_path

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

def plot_waveform(samples: torch.Tensor):
    import matplotlib.pyplot as plt
    import numpy as np

    sample_rate = 32000
    fig, ax = plt.subplots()
    ax.plot(np.linspace(0, len(samples[0]) / sample_rate, num=len(samples[0])), samples[0].numpy())
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Amplitude')
    st.pyplot(fig)

st.set_page_config(
    page_icon="🎵",
    page_title="Music Gen"
)

def main():
    st.title("Text to Music Generator🎵")

    with st.expander("See explanation"):
        st.write("Music Generator app built using Meta's Audiocraft library. We are using Music Gen Small model.")

    text_area = st.text_area("Enter your description.......")
    time_slider = st.slider("Select time duration (In Seconds)", 0, 20, 10)
    model_option = st.selectbox("Choose a model", ["facebook/musicgen-small", "facebook/musicgen-medium", "facebook/musicgen-large"])
    effect_option = st.selectbox("Choose an audio effect", ["None", "Reverb", "Echo"])

    if text_area and time_slider:
        st.json({
            'Your Description': text_area,
            'Selected Time Duration (in Seconds)': time_slider,
            'Selected Model': model_option,
            'Selected Effect': effect_option
        })

        st.subheader("Generated Music")
        music_tensors = generate_music_tensors(text_area, time_slider, model_option)
        if effect_option != "None":
            music_tensors = apply_audio_effects(music_tensors, effect_option)
        audio_filepath = save_audio(music_tensors)
        audio_file = open(audio_filepath, 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes)
        st.markdown(get_binary_file_downloader_html(audio_filepath, 'Audio'), unsafe_allow_html=True)

        st.subheader("Waveform")
        plot_waveform(music_tensors)

    st.sidebar.subheader("Save/Load Descriptions")
    if st.sidebar.button("Save Description"):
        with open("descriptions.json", "w") as f:
            json.dump({"description": text_area, "duration": time_slider}, f)
        st.sidebar.write("Description saved!")

    if st.sidebar.button("Load Description"):
        if os.path.exists("descriptions.json"):
            with open("descriptions.json", "r") as f:
                data = json.load(f)
            st.sidebar.write("Description loaded!")
            st.text_area("Loaded Description", data["description"], key="loaded_description")
            st.slider("Loaded Duration", 0, 20, data["duration"], key="loaded_duration")
        else:
            st.sidebar.write("No saved description found.")

if __name__ == "__main__":
    main()
