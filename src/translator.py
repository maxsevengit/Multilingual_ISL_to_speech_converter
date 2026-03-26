"""
ISL to Speech Translator Module.

Uses Google Gemini (Generative AI) to convert a list of isolated
ISL glosses (e.g., ["HELLO", "HOW_ARE_YOU"]) into fluent, grammatically
correct sentences in various Indian languages.

For demo reliability, this module supports an OFFLINE fallback:
  - Sentence formation: simple rule-based join + light cleanup
  - Speech: macOS `say` (no internet needed)
"""

import os
import threading
import subprocess
from dotenv import load_dotenv

# Load environment variables (API Key)
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

def _try_configure_gemini():
    if not api_key:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        return genai
    except Exception:
        return None


class ISLTranslator:
    def __init__(self):
        self.genai = _try_configure_gemini()
        self.is_configured = self.genai is not None
        self.model = None
        if self.is_configured:
            try:
                self.model = self.genai.GenerativeModel('gemini-2.5-flash')
            except Exception:
                self.is_configured = False
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
            "Bengali": "bn",
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
            translated_text = self._translate(glosses, target_lang)
            self.last_translation = translated_text

            # Offline speech (macOS)
            self._speak_macos(translated_text, target_lang)
                
        except Exception as e:
            print(f"[ERROR] Translation failed: {e}")
            self.last_translation = "Translation Failed!"
        finally:
            self.is_translating = False

    def _translate(self, glosses: list, target_lang: str) -> str:
        """
        Translate a list of gloss tokens to a fluent sentence.
        - If Gemini is available, use it.
        - Otherwise, use a deterministic offline fallback (demo-safe).
        """
        cleaned = [g.strip().replace("_", " ") for g in glosses if str(g).strip()]
        if not cleaned:
            return ""

        if self.is_configured and self.model is not None:
            gloss_str = ", ".join(cleaned)
            prompt = (
                f"You are an expert Indian Sign Language translator. "
                f"Take the following sequence of isolated sign language words (glosses): [{gloss_str}]. "
                f"Construct a single, fluent, naturally sounding, grammatically correct sentence representing their combined meaning. "
                f"Translate this final sentence into {target_lang}. "
                f"Respond ONLY with the translated {target_lang} sentence, nothing else."
            )
            response = self.model.generate_content(prompt)
            return (response.text or "").strip()

        # Offline fallback: simple gloss → sentence
        text = " ".join(cleaned)
        text = " ".join(text.split())
        # Light greeting-centric cleanup for nicer demo output
        replacements = {
            "HOW ARE YOU": "How are you",
            "GOOD MORNING": "Good morning",
            "GOOD AFTERNOON": "Good afternoon",
            "GOOD EVENING": "Good evening",
            "GOOD NIGHT": "Good night",
            "THANK YOU": "Thank you",
            "PLEASED": "Nice to meet you",
            "HELLO": "Hello",
            "ALRIGHT": "I'm fine",
            "SORRY": "Sorry",
        }
        upper = text.upper()
        if upper in replacements:
            text = replacements[upper]
        else:
            text = text.title()
        if not text.endswith((".", "!", "?")):
            text += "."
        return text

    def _speak_macos(self, text: str, target_lang: str):
        """
        Speak using macOS `say` (offline, reliable).
        For non-English languages, this will still speak but voice quality depends
        on installed system voices.
        """
        if not text:
            return
        try:
            subprocess.run(["say", text], check=False)
        except Exception:
            # Last resort: do nothing (keep UI responsive)
            return
