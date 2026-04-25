import base64
import os
import requests
import uuid
from fastapi import HTTPException, Depends
from sqlmodel import Session, select

from GenAnkiCards.task3_image_and_audio.dataclasses import ImageRequest, get_session, FlashcardDB, logger, slugify

# Images directory setup
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "./images")
os.makedirs(IMAGES_DIR, exist_ok=True)

# Dummy image paths
DUMMY_IMAGE_FS_PATH = os.path.join(IMAGES_DIR, "dummy.png")
DUMMY_IMAGE_URL = "./images/dummy.png"

try:
    import openai
    from GenAnkiCards.task2_flashcards.config import settings as task_settings

    _client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=task_settings.OPENROUTER_API_KEY
    ) if task_settings.OPENROUTER_API_KEY else None
except Exception:
    _client = None


def generate_image(req: ImageRequest, session: Session = Depends(get_session)):
    logger.info(
        f"Starting image generation for word: '{req.word}', language: '{req.language}', flashcard_id: {req.flashcard_id}")

    # Check if the image already exists in a database for this word/language combination
    existing_card_with_image = None
    if req.flashcard_id:
        # If flashcard_id is provided, check if this specific card already has an image
        card = session.get(FlashcardDB, req.flashcard_id)
        if card and card.image_path:
            logger.info(f"Flashcard {req.flashcard_id} already has an image: {card.image_path}")
            return {"image_path": card.image_path, "prompt": f"Existing image for '{req.word}'"}
    else:
        # Check if any card with the same word/language combination has an image
        stmt = select(FlashcardDB).where(
            FlashcardDB.word == req.word,
            FlashcardDB.language == req.language,
            FlashcardDB.image_path.isnot(None)
        ).limit(1)
        existing_card_with_image = session.exec(stmt).first()

        if existing_card_with_image:
            logger.info(
                f"Found existing image for word '{req.word}' in language '{req.language}': {existing_card_with_image.image_path}")
            return {"image_path": existing_card_with_image.image_path, "prompt": f"Existing image for '{req.word}'"}

    # No existing image found, proceed with generation
    prompt = (
        f"Create a complete, full-frame illustration of the word '{req.word}' "
        f"in the context of '{req.language}' language. "
        f"Ensure the entire subject is visible within the frame with adequate margins and padding. "
        f"Do not crop any important parts. Center the main subject with enough white space or background around it. "
        f"Make it a clean, educational illustration suitable for learning vocabulary."
    )
    logger.info(f"Generated prompt: {prompt}")

    safe_word = slugify(req.word, "image")
    unique_id = uuid.uuid4().hex[:12]

    # png is safe (supports transparency)
    output_format = "png"
    filename = f"{safe_word}_{unique_id}.{output_format}"

    image_fs_path = os.path.join(IMAGES_DIR, filename)

    try:
        if _client is None:
            logger.error("Openrouter client is not configured - API key missing or invalid")
            raise RuntimeError("Image client is not configured")

        logger.info("Calling Openrouter for image generation...")
        
        headers = {
            "Authorization": f"Bearer {task_settings.OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "openai/gpt-image-1",
            "messages": [{"role": "user", "content": prompt}],
        }
        
        response = requests.post("https://openrouter.ai/api/v1/chat/completions",
                               headers=headers, json=payload)
        result = response.json()
        
        # Extract image from the response
        if not result.get("choices"):
            raise RuntimeError("No choices returned from API")
            
        message = result["choices"][0]["message"]
        if not message.get("images"):
            raise RuntimeError("No images returned from API")
            
        # Get the first generated image
        image_data = message["images"][0]["image_url"]["url"]  # Base64 data URL
        
        # Extract base64 data from data URL format (data:image/png;base64,...)
        if image_data.startswith("data:"):
            # Split on comma to get the base64 part
            base64_data = image_data.split(",", 1)[1]
        else:
            base64_data = image_data
            
        with open(image_fs_path, "wb") as f:
            f.write(base64.b64decode(base64_data))
        logger.info(f"Image saved to local file system: {image_fs_path}")

        image_url_path = f"./images/{filename}"

    except Exception as e:
        logger.exception(f"Image generation failed, falling back to dummy image. Error: {e}")
        image_fs_path = DUMMY_IMAGE_FS_PATH
        image_url_path = DUMMY_IMAGE_URL

    # Persist to DB if flashcard_id provided: store relative web path in image_path
    if req.flashcard_id is not None:
        card = session.get(FlashcardDB, req.flashcard_id)
        if not card:
            raise HTTPException(status_code=404, detail="Flashcard not found for image update")
        card.image_path = image_url_path  # store relative path like './images/...'
        session.add(card)
        session.commit()
        session.refresh(card)

    logger.info(f"Generated image URL: {image_url_path}")

    logger.info(f"Image generation completed. File: {image_fs_path}, URL: {image_url_path}")
    return {"image_path": image_url_path, "prompt": prompt}
