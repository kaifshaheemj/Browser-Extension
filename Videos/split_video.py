# import os
# from moviepy.editor import VideoFileClip

# def split_video(video_path, chunk_length=60, output_folder="video_chunks"):
#     """Splits a long video into smaller clips of a given duration (default: 60 seconds)."""
    
#     # Create output folder if not exists
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     # Load the video without audio to avoid issues
#     video = VideoFileClip(video_path, audio=True)
    
#     duration = video.duration  # Get total duration in seconds
#     print(f"Total Video Duration: {duration} seconds")
    
#     # Calculate total number of chunks expected
#     total_chunks = (int(duration) + chunk_length - 1) // chunk_length
#     print(f"Expected chunks: {total_chunks}")

#     chunks = []
#     successful_chunks = 0
    
#     for i in range(total_chunks):
#         start_time = i * chunk_length
#         end_time = min(start_time + chunk_length, duration)
        
#         print(f"Processing chunk {i+1}/{total_chunks}: {start_time}s to {end_time}s")
        
#         # Extract subclip
#         clip = video.subclip(start_time, end_time)
        
#         # Get dynamic FPS
#         fps = video.fps if video.fps else 24  # Use original FPS or default to 24
        
#         chunk_path = os.path.join(output_folder, f"chunk_{i+1}.mp4")
        
#         try:
#             # Write without audio
#             clip.write_videofile(
#                 chunk_path, 
#                 codec="libx264", 
#                 fps=fps, 
#                 audio=True
#             )
                
#             chunks.append(chunk_path)
#             successful_chunks += 1
#             print(f"Created chunk {i+1}/{total_chunks}: {chunk_path}")
            
#         except Exception as e:
#             print(f"Failed to create chunk {i+1}/{total_chunks}: {e}")
        
#         # Close the clip to release memory
#         clip.close()

#     # Close the original video
#     video.close()
    
#     print(f"Successfully created {successful_chunks} out of {total_chunks} expected chunks")
#     return chunks

# # Example Usage
# video_path = r"D:\PySceneDetect\structure\videos\videos1.2 Characteristics of Algorithm.mp4"
# chunks = split_video(video_path)
# print("Video successfully split into:", chunks)

# import os
# import gc
# from moviepy.editor import VideoFileClip

# def split_video(video_path, chunk_length=60, output_folder="video_chunks"):
#     """Splits a video into chunks while retaining audio."""
    
#     # Create output folder if it doesn't exist
#     os.makedirs(output_folder, exist_ok=True)

#     # Load the video
#     try:
#         video = VideoFileClip(video_path)
#     except Exception as e:
#         print(f"Error loading video: {e}")
#         return []

#     duration = video.duration  # Total video duration
#     print(f"Total Video Duration: {duration:.2f} seconds")
    
#     # Calculate total number of chunks expected
#     total_chunks = (int(duration) + chunk_length - 1) // chunk_length
#     print(f"Expected chunks: {total_chunks}")

#     chunks = []
    
#     for i in range(total_chunks):
#         start_time = i * chunk_length
#         end_time = min(start_time + chunk_length, duration)
        
#         print(f"Processing chunk {i+1}/{total_chunks}: {start_time}s to {end_time}s")
        
#         # Extract subclip
#         clip = video.subclip(start_time, end_time)
        
#         chunk_path = os.path.join(output_folder, f"chunk_{i+1}.mp4")
        
#         try:
#             # Explicitly define audio codec to avoid NoneType error
#             clip.write_videofile(
#                 chunk_path, 
#                 codec="libx264", 
#                 fps=video.fps or 24, 
#                 audio_codec="aac",  # Ensure proper audio processing
#                 temp_audiofile=f"temp_audio_{i+1}.m4a"  # Unique temp file for audio
#             )
                
#             chunks.append(chunk_path)
#             print(f"Created chunk {i+1}/{total_chunks}: {chunk_path}")
            
#         except Exception as e:
#             print(f"Failed to create chunk {i+1}/{total_chunks}: {e}")
        
#         # Close the clip to release memory
#         clip.close()
#         del clip
#         gc.collect()  # Force garbage collection

#     # Close and delete original video
#     video.close()
#     del video
#     gc.collect()
    
#     print(f"Successfully created {len(chunks)} out of {total_chunks} expected chunks")
#     return chunks

# # Example Usage
# video_path = r"D:\PySceneDetect\structure\videos\videos1.2 Characteristics of Algorithm.mp4"
# chunks = split_video(video_path)
# print("Video successfully split into:", chunks)

# import os
# from scenedetect import SceneManager
# from scenedetect.detectors import ContentDetector
# from scenedetect.video_manager import VideoManager
# from scenedetect.scene_manager import save_images
# from scenedetect.frame_timecode import FrameTimecode

# def split_video(video_path, output_folder, threshold=30.0):
#     # Initialize Video Manager
#     video_manager = VideoManager([video_path])
#     scene_manager = SceneManager()
#     scene_manager.add_detector(ContentDetector(threshold=threshold))

#     # Start processing video
#     video_manager.set_downscale_factor()
#     video_manager.start()
#     scene_manager.detect_scenes(frame_source=video_manager)
#     scene_list = scene_manager.get_scene_list()

#     # Print detected scenes
#     print(f"Detected {len(scene_list)} scenes.")
#     for i, (start, end) in enumerate(scene_list):
#         print(f"Scene {i+1}: Start time: {start}, End time: {end}")

#     # Split video into individual scenes
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     for i, (start_time, end_time) in enumerate(scene_list):
#         start_frame = start_time.get_frames()
#         end_frame = end_time.get_frames()
#         scene_filename = os.path.join(output_folder, f"scene_{i+1}.mp4")

#         # Export scene using FFmpeg
#         os.system(f'ffmpeg -i "{video_path}" -vf "select=between(n\,{start_frame}\,{end_frame})" -vsync vfr "{scene_filename}"')

#     video_manager.release()
#     print(f"Scenes saved in: {output_folder}")

# # Example usage
# video_path = r"D:\\PySceneDetect\\structure\\videos\\videos1.2 Characteristics of Algorithm.mp4"
# output_folder = "output_scenes"

# split_video(video_path, output_folder)

# from scenedetect import open_video, SceneManager, split_video_ffmpeg
# from scenedetect.detectors import ContentDetector
# from scenedetect.video_splitter import split_video_ffmpeg

# def split_video_into_scenes(video_path, threshold=27.0):
#     # Open our video, create a scene manager, and add a detector.
#     video = open_video(video_path)
#     scene_manager = SceneManager()
#     scene_manager.add_detector(
#         ContentDetector(threshold=threshold))
#     scene_manager.detect_scenes(video, show_progress=True)
#     scene_list = scene_manager.get_scene_list()
#     split_video_ffmpeg(video_path, scene_list, show_progress=True)

# split_video_into_scenes(r"D:\PySceneDetect\structure\videos\videos1.1 Priori Analysis and Posteriori Testing.mp4")

# from scenedetect import open_video, SceneManager, ContentDetector
# video = open_video(r'D:\PySceneDetect\structure\videos\videos1.1 Priori Analysis and Posteriori Testing.mp4')
# scene_manager = SceneManager()
# scene_manager.add_detector(ContentDetector(threshold=27.0))
# scene_manager.detect_scenes(video)
# print(scene_manager.get_scene_list())

# import scenedetect

# # Import necessary methods from PySceneDetect
# from scenedetect import VideoManager, SceneManager
# from scenedetect.detectors import ContentDetector
# from scenedetect.video_splitter import split_video_ffmpeg

# # Input video file
# video_path = r"D:\PySceneDetect\structure\videos\videosInstagram-லயும் Maths-இருக்கா 🤔  Tamil  LMES #matrix #mathematics.mp4"

# # Create a video manager object
# video_manager = VideoManager([video_path])
# scene_manager = SceneManager()

# # Add a ContentDetector (with threshold of 30, adjust if needed)
# scene_manager.add_detector(ContentDetector(threshold=10.0))

# # Start video processing
# video_manager.start()

# # Detect scenes in the video
# scene_manager.detect_scenes(frame_source=video_manager)

# # Get the list of detected scenes
# scene_list = scene_manager.get_scene_list()

# # Print scene timecodes
# print("Detected scenes:", [(scene[0].get_timecode(), scene[1].get_timecode()) for scene in scene_list])

# # Split video using FFmpeg
# split_video_ffmpeg(video_path, scene_list, output_dir="output_scenes", show_output=True)

# # Release resources
# video_manager.release()

import cv2
import ffmpeg
import speech_recognition as sr
from moviepy.editor import VideoFileClip
from scenedetect import open_video, SceneManager, ContentDetector
from fpdf import FPDF
import os
# Directories
video_path = r"D:\PySceneDetect\structure\videos\videosYCombinator Demo video - Helpingai.mp4"
video_output_dir = r"D:\PySceneDetect\structure\videos"
audio_output_dir = r"D:\PySceneDetect\structure\audios"
frame_output_dir = r"D:\PySceneDetect\structure\frames"

# Ensure directories exist
os.makedirs(video_output_dir, exist_ok=True)
os.makedirs(audio_output_dir, exist_ok=True)
os.makedirs(frame_output_dir, exist_ok=True)

# Open video & detect scenes
video = open_video(video_path)
scene_manager = SceneManager()
scene_manager.add_detector(ContentDetector(threshold=10.0))

# Detect scenes
scene_manager.detect_scenes(video)
scene_list = scene_manager.get_scene_list()

print(f"Detected {len(scene_list)} scenes.")

# Initialize SpeechRecognition
recognizer = sr.Recognizer()

# Create PDF
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)

# Process each scene
for i, (start, end) in enumerate(scene_list):
    scene_filename = f"scene-{i+1:03d}.mp4"
    scene_path = os.path.join(video_output_dir, scene_filename)
    
    # Extract scene using FFmpeg
    (
        ffmpeg.input(video_path, ss=start.get_seconds(), to=end.get_seconds())
        .output(scene_path)
        .run(overwrite_output=True, quiet=True)
    )
    
    # Extract audio
    audio_filename = f"scene-{i+1:03d}.wav"
    audio_path = os.path.join(audio_output_dir, audio_filename)

    (
        ffmpeg.input(scene_path)
        .output(audio_path, acodec="pcm_s16le", ar="16000")
        .run(overwrite_output=True, quiet=True)
    )

    # Transcribe audio using SpeechRecognition
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        try:
            transcript = recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            transcript = "[Could not transcribe audio]"
        except sr.RequestError:
            transcript = "[Error connecting to speech recognition service]"

    # Extract first & last frames
    cap = cv2.VideoCapture(scene_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_times = [0, total_frames - 2] if total_frames > 2 else [0]
    frame_images = []
    
    for frame_no in frame_times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if ret:
            frame_file = os.path.join(frame_output_dir, f"frame_scene_{i+1:03d}_frame_{frame_no}.jpg")
            cv2.imwrite(frame_file, frame)
            frame_images.append(frame_file)

    cap.release()

    # Append to PDF
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"Scene {i+1}", ln=True, align='C')
    
    for img in frame_images:
        pdf.image(img, x=10, w=180)
    
    pdf.multi_cell(0, 10, f"Transcript:\n{transcript}")

# Save PDF
pdf_path = os.path.join(video_output_dir, "scenes_transcripts.pdf")
pdf.output(pdf_path)
print(f"PDF saved at: {pdf_path}")


