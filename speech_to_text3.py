import whisper
from moviepy.editor import AudioFileClip
import os
import torch

# Define paths
base_path = ""
file = input("Enter Filename:  ")
audio_path = os.path.join(base_path, file)
audio_partition_folder = os.path.join(base_path, "audio_files")
subtitle_path = os.path.join(base_path, "subtitles", "subtitles.txt")

# Ensure directories exist
os.makedirs(audio_partition_folder, exist_ok=True)
os.makedirs(os.path.dirname(subtitle_path), exist_ok=True)

# Load audio and get duration
audio_clip = AudioFileClip(audio_path)
n = round(audio_clip.duration)
audio_clip.close()

# Partitioning parameters
start = 0
index = 60  # 1-minute segments
counter = 0

print("Partitioning the audio clip...")

# Partition the audio into 1-minute segments
while start < n:
    end_time = min(start + index, n)

    # Extract and save subclip
    temp_saving_location = os.path.join(
        audio_partition_folder, f'temp_{counter}.mp3')
    with AudioFileClip(audio_path) as audio_clip:
        temp = audio_clip.subclip(start, end_time)
        temp.write_audiofile(filename=temp_saving_location,
                             verbose=False, logger=None)
        temp.close()

    start = end_time  # Move to next segment
    counter += 1

print("Partitioning completed.")

# Load Whisper model once
print("Loading Whisper model...")
model = whisper.load_model(
    "large", device='cuda' if torch.cuda.is_available() else 'cpu')

# Transcribing each segment
print("Transcribing audio segments...")
final_list_of_text = []
id_counter = 0
start_time = 0

for index in range(counter):
    path_to_saved_file = os.path.join(
        audio_partition_folder, f'temp_{index}.mp3')

    # Ensure the file exists
    if not os.path.exists(path_to_saved_file):
        continue

    with AudioFileClip(path_to_saved_file) as audio_clip:
        duration = audio_clip.duration

    # Transcribe using Whisper
    out = model.transcribe(path_to_saved_file)
    list_of_text = out['segments']

    for line in list_of_text:
        line['start'] += start_time
        line['end'] += start_time
        line['id'] = id_counter
        id_counter += 1
        final_list_of_text.append(line)

    start_time += duration  # Update timestamp

    # Delete temporary file to free memory
    os.remove(path_to_saved_file)

# Save subtitles
with open(subtitle_path, 'w') as fp:
    for line in final_list_of_text:
        fp.write("{}\n".format(line.get("text")))

print("Successfully completed execution.")
