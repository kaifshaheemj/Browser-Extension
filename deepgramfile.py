#DEEPGRAM CODE FOR SPEECH TO TEXT

import json
import os
from deepgram import DeepgramClient, PrerecordedOptions, FileSource


def transcribe_audio(audio_file_path: str) -> str:

    try:
        # Initialize Deepgram client
        deepgram = DeepgramClient("133c50a3d15f66f7795fbfdb6a795675509456dc")
        
        # Read the audio file as binary
        with open(audio_file_path, "rb") as file:
            buffer_data = file.read()
        
        payload: FileSource = {"buffer": buffer_data}
        options = PrerecordedOptions(model="base", smart_format=True)

        # Call Deepgram API for transcription
        response = deepgram.listen.rest.v("1").transcribe_file(payload, options)

        # Convert response to JSON
        transcript_json = response.to_json()
        transcript_data = json.loads(transcript_json)
        
        # Extract the transcript text
        transcript_text = transcript_data.get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("transcript", "")

        if not transcript_text:
            raise ValueError("No transcript found in Deepgram response.")

        return transcript_text
    
    except Exception as e:
        print(f"Error during transcription: {e}")
        return "Transcription failed."

# Example usage
if __name__ == "__main__":
    audio_path = "/Users/saivignesh/Documents/DGM_Project/Browser-Extension/WebAssistAI/audio/1743645512363w9vsqrwn-voicemaker.in-speech.wav"
    transcript = transcribe_audio(audio_path)
    print("Transcript:", transcript)