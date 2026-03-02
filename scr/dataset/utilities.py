from datasets import load_dataset
from torchcodec.decoders import AudioDecoder
import random
from pathlib import Path
import json
import numpy as np
from scipy.io import wavfile

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
    """
    Schema-robust text accessor.

    Supports:
      - local subset records: {"text": "..."}
      - HF streaming samples: {"json": {"text": "..."}} (common for Emilia)
      - HF samples with top-level "text"
    """
    text = sample.get("text", None)
    if isinstance(text, str) and text.strip():
        return text

    json_obj = sample.get("json", None)
    if isinstance(json_obj, dict):
        text2 = json_obj.get("text", None)
        if isinstance(text2, str) and text2.strip():
            return text2

    # If it failed, return empty text
    return ""


def extract_audio_np_and_sr(sample: dict) -> tuple[np.ndarray, int]:
    """
    Schema-robust audio accessor.

    Supports:
      - local subset records: {"wav": np.int16 array, "sr": int}
      - streaming samples where sample["mp3"] is a torchcodec AudioDecoder
      - non-streaming samples where sample["mp3"] is a dict: {"array": ..., "sampling_rate": ...}

    Returns:
      (wav_mono_float32, sr_int)
      - wav is mono float32 in roughly [-1, 1] (best-effort)
    """
    # Local subset schema
    if "wav" in sample and "sr" in sample:
        sr = int(sample["sr"])
        wav = np.asarray(sample["wav"])

        # PCM16 -> float32 [-1, 1]
        if wav.dtype == np.int16:
            wav = wav.astype(np.float32) / 32768.0
        else:
            wav = wav.astype(np.float32, copy=False)

        # Mono
        if wav.ndim == 2:
            # If shape is (samples, channels) average channels
            if wav.shape[0] >= wav.shape[1]:
                wav = wav.mean(axis=1)
            else:
                wav = wav.mean(axis=0)

        return wav, sr

    # HF schema: mp3
    if "mp3" not in sample:
        raise KeyError(f"Missing audio keys. Available keys: {list(sample.keys())}")

    mp3 = sample["mp3"]

    if isinstance(mp3, AudioDecoder):
        s = mp3.get_all_samples()
        sr = int(s.sample_rate)
        # torchcodec returns tensor shaped (channels, samples)
        wav = s.data.detach().cpu().numpy()
        if wav.ndim == 2:
            wav = wav.mean(axis=0)  # mono
        wav = wav.astype(np.float32, copy=False)
        return wav, sr

    if isinstance(mp3, dict) and "array" in mp3 and "sampling_rate" in mp3:
        sr = int(mp3["sampling_rate"])
        wav = np.asarray(mp3["array"], dtype=np.float32)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)  # common shape: (samples, channels)
        return wav, sr

    raise TypeError(f"Unsupported sample['mp3'] type: {type(mp3)!r}")


def download_emilia_subset(
    *,
    n: int,
    seed: int,
    out_dir: str | Path,
    data_files_path: str = EMILIA_PATH,
    overwrite: bool = False,
) -> Path:
    """
    Deterministically picks N samples by streaming + shuffle(seed=seed),
    then saves audio + manifest to out_dir.

    Re-running with the same seed + n is stable (as long as upstream dataset doesn't change).
    """
    out_dir = Path(out_dir)
    audio_dir = out_dir / "audio"
    manifest_path = out_dir / "manifest.jsonl"

    if out_dir.exists() and overwrite:
        # minimal "overwrite" behavior: remove manifest; audio files will be overwritten by name
        # (keeps it safer than deleting the whole folder accidentally)
        if manifest_path.exists():
            manifest_path.unlink()

    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    ds = load_emilia(path=data_files_path)
    ds = ds.shuffle(seed=int(seed))  # works with streaming IterableDataset

    written = 0
    with manifest_path.open("a", encoding="utf-8") as f:
        for idx, sample in enumerate(ds):
            if written >= n:
                break

            key = str(sample.get("__key__", f"idx{idx:06d}"))

            text = get_sample_text(sample)

            wav, sr = extract_audio_np_and_sr(sample)

            # safe filename: prefix with numeric index to keep stable ordering on disk
            wav_name = f"{written:06d}_{key}.wav"
            wav_path = audio_dir / wav_name

            # Write PCM16 for compatibility (smaller, fast to load)
            wav_i16 = np.clip(wav, -1.0, 1.0)
            wav_i16 = (wav_i16 * 32767.0).astype(np.int16)
            wavfile.write(wav_path, sr, wav_i16)

            rec = {
                "i": written,
                "key": key,
                "text": text,
                "sr": sr,
                "num_samples": int(wav.shape[0]),
                "path": str(wav_path.as_posix()),
                "url": sample.get("__url__", None),
                "seed": int(seed),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            written += 1
            if written % 25 == 0:
                print(f"Downloaded {written}/{n}...")

    if written < n:
        raise RuntimeError(f"Only downloaded {written} samples, expected {n}.")
    print(f"Done. Wrote {written} samples to: {out_dir}")
    return out_dir


def iter_local_subset(out_dir: str | Path):
    """
    Yields dicts with:
      - "wav": np.int16 array
      - "sr": int
      - "text": str
      - plus the original manifest fields
    """
    out_dir = Path(out_dir)
    manifest_path = out_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            sr, wav_i16 = wavfile.read(Path(rec["path"]))
            yield {
                **rec,
                "sr": int(sr),
                "wav": wav_i16,  # int16 PCM
            }


if __name__ == "__main__":
    # dataset = load_emilia()
    # print(dataset)
    #
    # sample: dict = next(iter(dataset))
    # print(sample)
    #
    # text = get_sample_text(sample)
    # print(f"Text: {text}")
    #
    # download_audio_file(sample, file_name="test_sample")
    # print("Saved test_sample.wav")

    dataset = load_emilia()
    random_seed = random.randint(0, 10000)
    shuffled_data = dataset.shuffle(seed=random_seed)
    sample = next(iter(shuffled_data))
    text = get_sample_text(sample)
    print(f"Random Text: {text}")
    download_audio_file(sample, file_name="random_sample")

    # Example usage of deterministic subset download:
    # download_emilia_subset(n=1000, seed=123, out_dir="emilia_subset_seed123_n1000", overwrite=False)

    # Example usage of local iterator:
    # for rec in iter_local_subset("emilia_subset_seed123_n1000"):
    #     print(rec["i"], rec["key"], rec["sr"], rec["text"][:60])
    #     break



# #region Order by Pitch

# import numpy as np

# def to_mono(audio):
#     if audio.ndim > 1:
#         return np.mean(audio, axis=1)
#     return audio

# import librosa
# import numpy as np

# def estimate_pitch(audio, sr):
#     audio = to_mono(audio)
#
#     # Extract pitch values over time
#     f0, voiced_flag, voiced_prob = librosa.pyin(
#         audio,
#         fmin=librosa.note_to_hz('C2'),  # lower bound of human speech # type: ignore
#         fmax=librosa.note_to_hz('C6'),  # upper bound # type: ignore
#         sr=sr
#     )
#
#     # Remove unvoiced frames (NaNs)
#     f0 = f0[~np.isnan(f0)]
#
#     if len(f0) == 0:
#         return 0  # fallback
#
#     return np.median(f0)  # robust against pitch variation


# print("Fetching audio samples")

# audio_samples = []
# iterable = iter(dataset.shuffle(seed=123))
# for i in range(10):
#     audio_samples.append(next(iterable))


# def pitch_of_sample(sample):
#     samples = sample['mp3'].get_all_samples()
#
#     # Extract tensor
#     waveform = samples.data        # <- this is the actual tensor
#     sample_rate = samples.sample_rate
#
#     #print(f"shape of waveform {waveform.shape}")
#
#     # Convert shape from (channels, samples) -> (samples, channels)
#     waveform = waveform.T
#
#     # Move to CPU and numpy
#     waveform = waveform.cpu().numpy()
#     audio = waveform
#     sr = sample_rate
#     return estimate_pitch(audio, sr)


# print("Computing pitch values")

# # Precompute pitches once (important for performance)
# for sample in audio_samples:
#     sample['pitch'] = pitch_of_sample(sample)


# print('Sorting samples')
# # Sort low → high pitch
# samples_sorted = sorted(audio_samples, key=lambda s: s['pitch'])


# print("Downloading sorted samples")

# # for i, sample in enumerate(samples_sorted):
# #     download_audio_file(sample, file_name=f"{i}")


# download_audio_file(samples_sorted[0], file_name=f"low")
# download_audio_file(samples_sorted[-1], file_name=f"high")

# #endregion
