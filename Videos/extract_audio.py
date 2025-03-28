import os
from moviepy.editor import VideoFileClip

def extract_audio(video_path, output_folder="audios"):
    """Extracts audio from the video and saves it as a WAV file."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    audio_path = os.path.join(output_folder, os.path.basename(video_path).replace(".mp4", ".wav"))

    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, codec='pcm_s16le')
    clip.close()  # Close the clip to free resources
    return audio_path

# Example Usage
video_path = r"D:\PySceneDetect\structure\video_chunks\chunk_1.mp4"
audio_file = extract_audio(video_path)
print(f"Extracted audio: {audio_file}")
