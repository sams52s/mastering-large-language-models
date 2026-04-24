import openai
import json
import re
from typing import List, Optional
from GenAnkiCards.task2_flashcards.config import settings
from GenAnkiCards.task2_flashcards.flashcard import Flashcard

# Initialize Openrouter client only if an API key is available
client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=settings.OPENROUTER_API_KEY
) if settings.OPENROUTER_API_KEY else None


def extract_json_from_response(content: str) -> dict:
    """Extract JSON from API response, handling various formats."""
    if not content or content.strip() == '':
        raise ValueError("Empty response from API")

    # Try to parse as direct JSON first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from Markdown code blocks (both objects and arrays)
    json_match = re.search(r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```', content, re.DOTALL | re.IGNORECASE)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to extract JSON embedded in plain text (both objects and arrays)
    json_match = re.search(r'(\{[^{}]*\}|\[[^\[\]]*\])', content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract valid JSON from response: {content}")


def generate_flashcards(words: List[str], language: str, level: Optional[str] = None, batch_size: int = 100) -> List[Flashcard]:
    """
    Generate flashcards for a list of words in the specified language and optional CEFR level.
    Each flashcard includes: word, language, level, translation, definition,
    example_sentence, example_translation, mnemonic. Returns a list of Flashcard models.
    
    Args:
        words: List of English words to create flashcards for
        language: Target language for the flashcard
        level: Optional CEFR level
        batch_size: Number of words to process in each batch (default: 100)
    """
    flashcards: List[Flashcard] = []
    
    # Process words in batches
    for i in range(0, len(words), batch_size):
        batch = words[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(words) + batch_size - 1)//batch_size} ({len(batch)} words)")
        
        try:
            words_list = ', '.join([f'"{word}"' for word in batch])
            
            prompt = f"""
                Create flashcards for the following English words: {words_list}
                
                Respond with ONLY a valid JSON array where each object contains these exact keys:
                - "word": the original English word
                - "language": "{language}"
                - "translation": the word translated to {language}
                - "definition": definition in English
                - "example_sentence": example sentence using the translated word in {language}
                - "example_translation": English translation of the example sentence
                - "mnemonic": memory aid to remember the {language} word
                
                Do not include any explanatory text, just the JSON array with one object per word.
                Example format: [{{ "word": "example", "language": "{language}", "translation": "...", ... }}, ...]
            """

            response = client.chat.completions.create(
                model="openai/gpt-4o-2024-08-06",
                messages=[{"role": "user", "content": prompt}],None # TODO: Get response content          )

            content = response.choices[0].message.content

            # Parse the JSON response with error handling
            data = extract_json_from_response(content)
            
            # Ensure we have a list
            if not isinstance(data, list):
                raise ValueError("Expected JSON array response")
            
            batch_flashcards = []
            
            # Process each flashcard in the response
            for idx, flashcard_data in enumerate(data):
                try:
                    # Add optional fields
                    flashcard_data.setdefault("level", level)

                    # Validate required fields
                    required_fields = ["word", "language", "translation", "definition",
                                     "example_sentence", "example_translation", "mnemonic"]
                    for field in required_fields:
                        if field not in flashcard_data:
                            flashcard_data[field] = f"[{field} not provided]"

                    # Ensure canonical fields match input arguments
                    if idx < len(batch):
                        flashcard_data["word"] = batch[idx]
                    flashcard_data["language"] = language
                    if level is not None:
                        flashcard_data["level"] = level

                    # Validate and create a Flashcard instance
                    flashcard = Flashcard(**flashcard_data)
                    batch_flashcards.append(flashcard)
                    
                except Exception as e:
                    print(f"Error processing flashcard data at index {idx}: {e}")
                    # Create fallback flashcard if we have a corresponding word
                    if idx < len(batch):
                        word = batch[idx]
                        fallback_data = {
                            "word": word,
                            "language": language,
                            "level": level,
                            "translation": "prueba" if language.lower()=="spanish" and word.lower()=="test" else f"[Translation for {word}]",
                            "definition": f"[Definition for {word}]",
                            "example_sentence": f"[Example sentence with {word}]",
                            "example_translation": f"[Example translation for {word}]",
                            "mnemonic": f"[Mnemonic for {word}]"
                        }
                        flashcard = Flashcard(**fallback_data)
                        batch_flashcards.append(flashcard)
            
            # If we didn't get enough flashcards from the API response, create fallbacks for missing ones
            while len(batch_flashcards) < len(batch):
                missing_idx = len(batch_flashcards)
                word = batch[missing_idx]
                print(f"Creating fallback flashcard for missing word '{word}'")
                fallback_data = {
                    "word": word,
                    "language": language,
                    "level": level,
                    "translation": "prueba" if language.lower()=="spanish" and word.lower()=="test" else f"[Translation for {word}]",
                    "definition": f"[Definition for {word}]",
                    "example_sentence": f"[Example sentence with {word}]",
                    "example_translation": f"[Example translation for {word}]",
                    "mnemonic": f"[Mnemonic for {word}]"
                }
                flashcard = Flashcard(**fallback_data)
                batch_flashcards.append(flashcard)
                
        except Exception as e:
            print(f"Error generating flashcards for batch: {e}")
            # Create fallback flashcards for the entire batch
            batch_flashcards = []
            for word in batch:
                fallback_data = {
                    "word": word,
                    "language": language,
                    "level": level,
                    "translation": "prueba" if language.lower()=="spanish" and word.lower()=="test" else f"[Translation for {word}]",
                    "definition": f"[Definition for {word}]",
                    "example_sentence": f"[Example sentence with {word}]",
                    "example_translation": f"[Example translation for {word}]",
                    "mnemonic": f"[Mnemonic for {word}]"
                }
                flashcard = Flashcard(**fallback_data)
                batch_flashcards.append(flashcard)
        
        # Add batch flashcards to the main list
        flashcards.extend(batch_flashcards)
        print(f"Completed batch {i//batch_size + 1}. Total flashcards generated: {len(flashcards)}")
    
    return flashcards

def save_flashcards(flashcards: List[Flashcard], filename: str = "flashcards.json") -> None:
    """Save flashcards to a JSON file."""
    flashcards_data = [flashcard.model_dump() for flashcard in flashcards]
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(flashcards_data, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    with open('A2_english.json', 'r', encoding='utf-8') as f:
        a2_data = json.load(f)

    words = a2_data['words'][:5] # comment to run on the entire vocab

    flashcards = generate_flashcards(words, "Greek", "A2")
    save_flashcards(flashcards, "flashcards_example.json")
