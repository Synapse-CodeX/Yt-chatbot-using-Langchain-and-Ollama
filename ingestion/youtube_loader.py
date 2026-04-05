from youtube_transcript_api import YouTubeTranscriptApi

def load_transcript(video_id: str) -> str:
    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.fetch(video_id)
        transcript = " ".join(chunk.text for chunk in transcript_list)
        return transcript
    except Exception as e:
        print("Error loading transcript:", e)
        raise RuntimeError(f"Failed to load transcript for video: {video_id}") from e