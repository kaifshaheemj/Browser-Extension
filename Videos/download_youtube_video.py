import os 
from pytubefix import YouTube
from pytubefix.cli import on_progress

def download_youtube_video(url, output_path= 'videos'):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    yt = YouTube(url, on_progress_callback = on_progress)
    video_path = os.path.join(output_path, yt.title + ".mp4")
    print(yt.title)
    ys = yt.streams.get_highest_resolution()
    ys.download(output_path=output_path, filename=video_path)

    return video_path

# download_youtube_video("https://youtu.be/-JTq1BFBwmo?si=shW0onVyHNdTmCB0")
download_youtube_video("https://youtu.be/VOOwIv73ZQo?si=Ja2aeeHZyHiTy1qP")
