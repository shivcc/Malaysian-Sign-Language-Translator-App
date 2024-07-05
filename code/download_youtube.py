from pytube import YouTube

# Replace 'YOUR_VIDEO_URL' with the URL of the YouTube video you want to download.
video_url = 'https://www.youtube.com/watch?v=ncGliuvNTT8'

# Replace 'YOUR_DOWNLOAD_PATH' with your preferred download location.
download_path = 'A:/Project-Sign-Language/ML-Raw-videos/Out_source/97'
filename = '2.mp4'

try:
    yt = YouTube(video_url)
    stream = yt.streams.get_highest_resolution()  # You can change this to get a different resolution.
    print(f'Downloading: {yt.title}...')
    stream.download(output_path=download_path ,filename=filename)
    print(f'Download complete! Video saved at: {download_path}')

except Exception as e:
    print(f'An error occurred: {str(e)}')
