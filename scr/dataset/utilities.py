from datasets import load_dataset
from torchcodec.decoders import AudioDecoder

EMILIA_PATH = "Emilia/EN/*.tar"


def load_emilia(path=EMILIA_PATH):
    return load_dataset(
        "amphion/Emilia-Dataset",
        data_files={"en": path},
        split="en",
        streaming=True,
    )


def download_audio_file(sample, file_name):
    if file_name is None:
        raise ValueError("file_name was not provided")

    import soundfile as sf

    decoder: AudioDecoder = sample["mp3"]
    samples = decoder.get_all_samples()

    waveform = samples.data
    sample_rate = samples.sample_rate

    waveform = waveform.T.cpu().numpy()
    sf.write(file_name + ".wav", waveform, sample_rate)


def get_sample_text(sample):
    return sample["json"]["text"]



def order_by_pitch_example():
    print("Order by pitch example\n")

    import numpy as np

    def to_mono(audio):
        if audio.ndim > 1:
            return np.mean(audio, axis=1)
        return audio

    import librosa
    import numpy as np

    def estimate_pitch(audio, sr):
        audio = to_mono(audio)

        # Extract pitch values over time
        f0, voiced_flag, voiced_prob = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),  # lower bound of human speech # type: ignore
            fmax=librosa.note_to_hz('C6'),  # upper bound # type: ignore
            sr=sr
        )

        # Remove unvoiced frames (NaNs)
        f0 = f0[~np.isnan(f0)]

        if len(f0) == 0:
            return 0  # fallback

        return np.median(f0)  # robust against pitch variation


    print("Fetching audio samples")

    audio_samples = []
    iterable = iter(dataset.shuffle(seed=100))
    for i in range(10):
        audio_samples.append(next(iterable))


    def pitch_of_sample(sample):
        samples = sample['mp3'].get_all_samples()

        # Extract tensor
        waveform = samples.data        # <- this is the actual tensor
        sample_rate = samples.sample_rate

        #print(f"shape of waveform {waveform.shape}")

        # Convert shape from (channels, samples) -> (samples, channels)
        waveform = waveform.T

        # Move to CPU and numpy
        waveform = waveform.cpu().numpy()
        audio = waveform
        sr = sample_rate
        return estimate_pitch(audio, sr)


    print("Computing pitch values")

    # Precompute pitches once (important for performance)
    for sample in audio_samples:
        sample['pitch'] = pitch_of_sample(sample)


    print('Sorting samples')
    # Sort low → high pitch
    samples_sorted = sorted(audio_samples, key=lambda s: s['pitch'])


    print("Downloading sorted samples")

    # for i, sample in enumerate(samples_sorted):
    #     download_audio_file(sample, file_name=f"{i}")


    download_audio_file(samples_sorted[0], file_name=f"low")
    download_audio_file(samples_sorted[-1], file_name=f"high")



if __name__ == "__main__":
    dataset = load_emilia()
    print(dataset)

    # sample: dict = next(iter(dataset))
    # print(sample)

    # text = get_sample_text(sample)
    # print(f"Text: {text}")

    # download_audio_file(sample, file_name="test_sample")
    # print("Saved test_sample.wav")

    order_by_pitch_example()
    

