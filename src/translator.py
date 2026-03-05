"""
ISL to Speech Translator Module.

Uses Google Gemini (Generative AI) to convert a list of isolated
ISL glosses (e.g., ["HELLO", "HOW_ARE_YOU"]) into fluent, grammatically
correct sentences in various Indian languages.
Uses Google Text-to-Speech (gTTS) to read the translated sentence aloud.
"""

import os
import threading
import subprocess
import google.generativeai as genai
from gtts import gTTS
from dotenv import load_dotenv

# Load environment variables (API Key)
load_dotenv()

# Configure Gemini
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)


class ISLTranslator:
    def __init__(self):
        self.is_configured = api_key is not None
        if self.is_configured:
            self.model = genai.GenerativeModel('gemini-2.5-flash')
        else:
            print("[WARNING] GEMINI_API_KEY not found in .env. Translation disabled.")
            self.model = None

        self.languages = {
            "English": "en",
            "Hindi": "hi",
            "Marathi": "mr",
            "Telugu": "te",
            "Tamil": "ta",
            "Gujarati": "gu",
            "Kannada": "kn",
            "Malayalam": "ml",
            "Bengali": "bn"
        }
        
        self.language_names = list(self.languages.keys())
        self.current_lang_idx = 0
        
        # State tracking for async translation
        self.is_translating = False
        self.last_translation = ""
        self.last_language = ""

    def get_current_language(self):
        return self.language_names[self.current_lang_idx]
        
    def next_language(self):
        self.current_lang_idx = (self.current_lang_idx + 1) % len(self.language_names)
        return self.get_current_language()

    def translate_and_speak_async(self, glosses: list):
        """Starts translation in a background thread so the video doesn't freeze."""
        if not self.is_configured:
            self.last_translation = "Error: GEMINI_API_KEY missing"
            return
            
        if not glosses or self.is_translating:
            return
            
        self.is_translating = True
        self.last_translation = "Translating..."
        target_lang = self.get_current_language()
        self.last_language = target_lang
        
        thread = threading.Thread(
            target=self._process_translation, 
            args=(glosses, target_lang)
        )
        thread.daemon = True
        thread.start()

    def _process_translation(self, glosses: list, target_lang: str):
        try:
            # 1. LLM Sentence Formation & Translation
            gloss_str = ", ".join(glosses)
            prompt = (
                f"You are an expert Indian Sign Language translator. "
                f"Take the following sequence of isolated sign language words (glosses): [{gloss_str}]. "
                f"Construct a single, fluent, and naturally sounding grammatically correct sentence representing their combined meaning. "
                f"Translate this final sentence into {target_lang}. "
                f"Respond ONLY with the translated {target_lang} sentence, nothing else. No quotes, no markdown."
            )
            
            response = self.model.generate_content(prompt)
            translated_text = response.text.strip()
            self.last_translation = translated_text
            
            # 2. Text-to-Speech
            lang_code = self.languages.get(target_lang, "en")
            tts = gTTS(text=translated_text, lang=lang_code, slow=False)
            
            audio_file = "temp_output.mp3"
            tts.save(audio_file)
            
            # 3. Audio Playback (macOS native)
            subprocess.run(["afplay", audio_file])
            
            # Cleanup
            if os.path.exists(audio_file):
                os.remove(audio_file)
                
        except Exception as e:
            print(f"[ERROR] Translation failed: {e}")
            self.last_translation = "Translation Failed!"
        finally:
            self.is_translating = False
