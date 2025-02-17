import warnings
import logging
import os
import gtts
from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from gtts import gTTS
import pygame  # For playing audio

# Disable TensorFlow OneDNN optimizations (optional, for certain hardware)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Suppress warnings and unnecessary logs
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Load environment variables
load_dotenv(find_dotenv())

# Define image-to-text function
def img2text(image_path):
    try:
        image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base", use_fast=True)
        text = image_to_text(image_path)[0]["generated_text"]
        return text
    except Exception as e:
        return f"Error in img2text: {e}"

# Define text-generation function
def generate_story(scenario):
    try:
        pipe = pipeline("text-generation", model="gpt2")
        story = pipe(scenario, max_length=100, do_sample=True)[0]['generated_text']
        return story
    except Exception as e:
        return f"Error in generate_story: {e}"

# Convert text to speech and save as an audio file
def text_to_speech(text, audio_path="story_audio.mp3"):
    try:
        tts = gTTS(text, lang="en")
        tts.save(audio_path)
        return audio_path
    except Exception as e:
        return f"Error in text_to_speech: {e}"

# Play audio
def play_audio(audio_path):
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()
        
        # Keep script running while audio plays
        while pygame.mixer.music.get_busy():
            continue
    except Exception as e:
        print(f"Error in play_audio: {e}")

# Process the image and generate a story
image_path = "./image.png"
scenario_text = img2text(image_path)

if scenario_text and not scenario_text.startswith("Error"):
    story = generate_story(scenario_text)
    
    print("\nGenerated Caption:", scenario_text)
    print("\nGenerated Story:", story)

    # Generate audio
    audio_file = text_to_speech(story)

    if not audio_file.startswith("Error"):
        print("\nPlaying the generated story...")
        play_audio(audio_file)
    else:
        print("\nFailed to generate audio.")
else:
    print("\nFailed to generate caption. Please check the image file.")
