import torch
import torchaudio
import os
from speechbrain.pretrained import EncoderClassifier
from collections import defaultdict

# Load pretrained speaker embedding model (ECAPA-TDNN)
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)


# Load your audio file
signal, fs = torchaudio.load("file_example_WAV_1MG.wav")

# Ensure correct sample rate (model expects 16kHz)
if fs != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
    signal = resampler(signal)

# Convert to mono if stereo
if signal.shape[0] > 1:
    signal = torch.mean(signal, dim=0, keepdim=True)

# Extract embedding
embedding = classifier.encode_batch(signal)

print("Embedding shape:", embedding.shape)

