import os
import json
import math
import subprocess

def split_and_convert_wav(input_file, target_size_mb=10):
    """
    Calculates the exact duration for a 10MB chunk based on the audio bitrate,
    then converts and splits the WAV file into FLAC chunks.
    """
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        return

    print("Analyzing input file properties...")
    
    # Use ffprobe to get the exact audio bitrate in bits per second
    probe_cmd = [
        'ffprobe', '-v', 'error', 
        '-show_entries', 'format=bit_rate', 
        '-of', 'json', input_file
    ]
    
    try:
        result = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        metadata = json.loads(result.stdout)
        # Fallback default if bitrate metadata is missing (1411200 bps for CD quality WAV)
        bitrate = int(metadata.get('format', {}).get('bit_rate', 1411200))
    except Exception as e:
        print(f"Warning: Could not read bitrate automatically ({e}). Using standard CD-quality default.")
        bitrate = 1411200 

    # Calculate target size in bits (10 MB * 1024 * 1024 * 8)
    target_bits = target_size_mb * 1024 * 1024 * 8
    
    # Calculate exact duration in seconds for uncompressed audio to fit 10MB
    # Because FLAC introduces compression (often 40-50%), this acts as a safe upper bound.
    duration_seconds = math.floor(target_bits / bitrate)
    
    print(f"Detected bitrate: {bitrate / 1000:.1f} kbps")
    print(f"Calculated target chunk duration: {duration_seconds} seconds")
    print("Converting and splitting into FLAC chunks...")

    # Base output name without extension
    base_name, _ = os.path.splitext(input_file)
    output_pattern = f"{base_name}_part_%03d.flac"

    # FFmpeg command to segment the audio and encode to FLAC on the fly
    ffmpeg_cmd = [
        'ffmpeg', '-i', input_file,
        '-f', 'segment',
        '-segment_time', str(duration_seconds),
        '-c:a', 'flac',
        output_pattern
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print("\nProcess complete! Your FLAC chunks have been successfully generated.")
    except subprocess.CalledProcessError as e:
        print(f"\nAn error occurred during FFmpeg execution: {e}")

if __name__ == "__main__":
    # Replace with the path to your actual WAV file
    your_wav_file = "audio.wav" 
    split_and_convert_wav(your_wav_file)

