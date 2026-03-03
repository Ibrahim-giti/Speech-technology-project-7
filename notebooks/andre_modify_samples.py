import parselmouth
from parselmouth.praat import call


def modify_f0_mean(input_path, output_path, factor):
    """
    Modifies the f0 mean of an audio file.
    factor > 1.0 increases pitch (e.g., 1.5 is a 50% increase)
    factor < 1.0 decreases pitch (e.g., 0.5 is an octave lower)
    """
    # 1. Load the audio file
    sound = parselmouth.Sound(input_path)

    # 2. Extract the manipulation object (uses PSOLA)
    # This identifies the pitch pulses in the original audio
    manipulation = call(sound, "To Manipulation", 0.01, 75, 600)

    # 3. Extract the Pitch Tier
    pitch_tier = call(manipulation, "Extract pitch tier")

    # 4. Modify the Pitch Tier by the desired factor
    # This multiplies every frequency point by your factor
    call(pitch_tier, "Multiply frequencies", sound.xmin, sound.xmax, factor)

    # 5. Replace the original pitch tier with the modified one
    call([manipulation, pitch_tier], "Replace pitch tier")

    # 6. Resynthesize the sound
    modified_sound = call(manipulation, "Get resynthesis (overlap-add)")

    # 7. Save the output
    modified_sound.save(output_path, "WAV")


# Example: Increase f0 mean by 50% (factor of 1.5)
modify_f0_mean("test_manipulate_original.wav", "output_increased_f0.wav", 1.5)