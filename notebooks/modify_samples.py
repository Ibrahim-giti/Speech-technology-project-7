import parselmouth
from parselmouth.praat import call


def modify_f0_mean(sound, pitch_factor):
    """
    Modifies the f0 mean of an audio file.
    pitch_factor > 1.0 increases pitch (e.g., 1.5 is a 50% increase)
    pitch_factor < 1.0 decreases pitch (e.g., 0.5 is an octave lower)
    """
    manipulation = call(sound, "To Manipulation", 0.01, 75, 600)
    pitch_tier = call(manipulation, "Extract pitch tier")
    call(pitch_tier, "Multiply frequencies", sound.xmin, sound.xmax, pitch_factor)
    call([manipulation, pitch_tier], "Replace pitch tier")
    new_sound = call(manipulation, "Get resynthesis (overlap-add)")
    return new_sound


def transform_voice(sound, pitch_factor, formant_factor):
    # Params:   call(sound, pitch_floor, pitch_ceil, formant_shift, pitch_median, pitch_range, duration)
    new_sound = call(sound, "Change speaker", 75, 600, formant_factor, pitch_factor, 1.0, 1.0)
    return new_sound


############################################################

original_sound = parselmouth.Sound("test_manipulate_original.wav")
factors = [0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00]

#Simple way
for f in factors:
    modified = modify_f0_mean(original_sound, f)
    modified.save(f"modified_pitch_simple/pitch_{f:4.2f}.wav", "WAV")

# Advanced way
for f in factors:
    # Heuristic: Formants usually move roughly 70% as much as the pitch for realism
    f_shift = 1.0 + (f - 1.0) * 0.7
    modified = transform_voice(original_sound, pitch_factor=f, formant_factor=f_shift)
    modified.save(f"modified_pitch_advanced/pitch_{f:4.2f}.wav", "WAV")


