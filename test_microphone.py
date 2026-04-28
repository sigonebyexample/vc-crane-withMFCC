import sounddevice as sd
import numpy as np
import time
import sys

print("=" * 60)
print("Microphone and Audio System Test")
print("=" * 60)

print("\nAvailable audio devices on system:")
print("-" * 60)
devices = sd.query_devices()
for i, dev in enumerate(devices):
    is_default_input = sd.default.device[0] == i
    is_default_output = sd.default.device[1] == i
    marker = ""
    if is_default_input:
        marker = " [Default Input]"
    elif is_default_output:
        marker = " [Default Output]"
    print(f"{i}: {dev['name']}")
    print(f"   Input channels: {dev['max_input_channels']}, Output channels: {dev['max_output_channels']}{marker}")
    print()

default_input = sd.default.device[0]
default_output = sd.default.device[1]
print(f"\nDefault input device: {devices[default_input]['name']}")
print(f"Default output device: {devices[default_output]['name']}")

print("\n" + "=" * 60)
print("Input Level Test (VU Meter)")
print("-" * 60)
print("Speak to see the volume level...")
print("Test duration: 5 seconds")
print("VU bar: [---] = silence, [###] = loud")
print()

def vu_meter(energy, max_energy=0.1):
    """Display volume level as a bar"""
    level = min(energy / max_energy, 1.0)
    bar_len = 30
    filled = int(level * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)
    return f"[{bar}] {energy:.4f}"

duration = 5
sample_rate = 16000
block_size = 1024

try:
    with sd.InputStream(samplerate=sample_rate, channels=1, blocksize=block_size) as stream:
        for _ in range(int(duration * sample_rate / block_size)):
            data, _ = stream.read(block_size)
            energy = np.sqrt(np.mean(data**2))
            print(f"\r{vu_meter(energy)}", end="", flush=True)
            time.sleep(0.05)
    print("\n\nLevel test completed.")
except Exception as e:
    print(f"\nError in audio test: {e}")
    print("Make sure the microphone is connected and enabled in Windows settings.")
    sys.exit(1)

print("\n" + "=" * 60)
print("Recording 3-second test file")
print("-" * 60)
print("Recording starts... say a short phrase (e.g., 'microphone test')")
time.sleep(1)

try:
    audio_test = sd.rec(int(3 * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    print("Recording completed.")

    energy = np.sqrt(np.mean(audio_test**2))
    print(f"Recorded file energy: {energy:.4f}")

    if energy < 0.002:
        print("\nWarning: Recorded file energy is very low!")
        print("   The microphone may be disconnected or your voice was not recorded.")
        print("   Please check the following:")
        print("   1. Microphone is enabled in Windows (Settings > System > Sound > Input)")
        print("   2. Microphone volume is set to 100%")
        print("   3. Physical microphone is not disconnected (on laptops)")
    else:
        print(f"Audio level is good (>{0.002:.4f})")

    print("\nPlaying back the recorded file...")
    sd.play(audio_test, samplerate=sample_rate)
    sd.wait()
    print("Playback completed. Did you hear your voice?")

except Exception as e:
    print(f"\nError in recording/playback: {e}")

print("\n" + "=" * 60)
print("Summary and Recommendations")
print("-" * 60)

input_device = devices[default_input]
if input_device['max_input_channels'] == 0:
    print("Default input device has no audio channel!")
    print("   Please set a microphone as the default input device.")
elif energy < 0.002:
    print("Microphone detected but recorded audio is weak.")
    print("   Check microphone settings in Windows:")
    print("   Settings > System > Sound > Input > Device Properties > Volume")
else:
    print("Everything looks great!")
    print("   Your microphone is ready for the voice command robot project.")
    print("\nYou can now run voice_robot_realtime_sklearn.py")

input("\nPress Enter to exit...")
