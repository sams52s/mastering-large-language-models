#!/usr/bin/env python3
"""
Upload flashcards from the local database (task3) to Anki via AnkiConnect.

- Reads flashcards from the SQLite DB defined in GenAnkiCards.task3_image_and_audio.dataclasses
- Uploads associated image and audio files as Anki media
- Adds notes to the requested deck using AnkiConnect (default model: Basic)

Requirements:
- Anki running with the AnkiConnect add-on enabled (default port 8765)
- The DB and media should come from task3_image_and_audio (relative paths like ./images/... and ./audio/...)
"""

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import click
import requests
from sqlmodel import Session, select
from tqdm import tqdm

from GenAnkiCards.task3_image_and_audio.dataclasses import (
    FlashcardDB,
    engine,
    DB_PATH,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("task4_anki_uploader")


# -------------------- AnkiConnect minimal client --------------------
class AnkiConnect:
    def __init__(self, endpoint: str = "http://127.0.0.1:8765"):
        self.endpoint = endpoint.rstrip("/")

    def _request(self, action: str, params: Optional[dict] = None):
        payload = {"action": action, "version": 6}
        if params:
            payload["params"] = params
        resp = requests.post(self.endpoint, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if data.get("error"):
            raise RuntimeError(f"AnkiConnect error in {action}: {data['error']}")
        return data.get("result")

    def create_deck(self, deck_name: str):
        return self._request("createDeck", {"deck": deck_name})

    def store_media_file(self, filename: str, path: Optional[str] = None, data_b64: Optional[str] = None):
        params = {"filename": filename}
        if path:
            params["path"] = path
        if data_b64:
            params["data"] = data_b64
        return self._request("storeMediaFile", params)

    def add_notes(self, notes: List[dict]):
        return self._request("addNotes", {"notes": notes})


# -------------------- Helpers --------------------
TASK3_DIR = Path(__file__).resolve().parent.parent / "task3_image_and_audio"
IMAGES_DIR = TASK3_DIR / "images"
AUDIO_DIR = TASK3_DIR / "audio"


def resolve_media_fs_path(rel_path: Optional[str]) -> Optional[str]:
    """Convert a DB relative path like './images/abc.png' to an absolute FS path."""
    if not rel_path:
        return None
    rp = rel_path.strip()
    if rp.startswith("./"):
        rp = rp[2:]
    abs_path = TASK3_DIR / rp
    return str(abs_path) if abs_path.exists() else None


@dataclass
class NoteBuild:
    front: str
    back: str
    image_filename: Optional[str] = None
    audio_filename: Optional[str] = None
    image_fs_path: Optional[str] = None
    audio_fs_path: Optional[str] = None


def build_note_from_record(card: FlashcardDB) -> NoteBuild:
    # Front and Back content
    front_parts = [
        f"{card.translation}",  # e.g., Greek translation
    ]
    if card.example_sentence:
        front_parts.append(card.example_sentence)

    # Back includes target word, example_translation, mnemonic, and media references
    back_parts = [
        *front_parts,
        f"<b>{card.word}</b>",
    ]
# TODO: add example_translation, definition and mnemonic to Anki card

    image_fs_path = resolve_media_fs_path(card.image_path)
    audio_fs_path = resolve_media_fs_path(card.audio_path)

    image_tag = ""
    audio_tag = ""

    image_filename = None
    audio_filename = None

    if image_fs_path:
        image_filename = os.path.basename(image_fs_path)
        image_tag = f'<div><img src="{image_filename}"></div>'
        back_parts.append(image_tag)

    if audio_fs_path:
        audio_filename = os.path.basename(audio_fs_path)
        audio_tag = f"[sound:{audio_filename}]"
        back_parts.append(audio_tag)

    front_html = "<br>".join([p for p in front_parts if p])
    back_html = "<br><br>".join([p for p in back_parts if p])

    return NoteBuild(
        front=front_html,
        back=back_html,
        image_filename=image_filename,
        audio_filename=audio_filename,
        image_fs_path=image_fs_path,
        audio_fs_path=audio_fs_path,
    )


# -------------------- Main CLI --------------------
@click.command()
@click.option("--deck", "deck_name", type=str, default="GreekCustom", help="Anki deck name")
@click.option("--limit", type=int, default=None, help="Limit number of cards to upload")
@click.option("--dry-run", is_flag=True, help="Do not call AnkiConnect; just show what would be uploaded")
@click.option("--model", "model_name", type=str, default="Basic", help="Anki note type (model) name")
@click.option("--ankiconnect", "anki_url", type=str, default="http://127.0.0.1:8765", help="AnkiConnect URL")
@click.option("--where", "where_expr", type=str, default=None, help="Optional SQL WHERE filter, e.g. language='Greek'")
def main(deck_name: str, limit: Optional[int], dry_run: bool, model_name: str, anki_url: str, where_expr: Optional[str]):
    logger.info(f"DB path: {DB_PATH}")
    logger.info(f"Deck: {deck_name} | Model: {model_name} | Dry-run: {dry_run}")

    # Load cards from DB
    with Session(engine) as session:
        stmt = select(FlashcardDB)
        # Only select cards not yet uploaded to Anki (uploaded_to_anki is False or NULL)
        try:
            # Use a simple equality; NULLs won't match, so include them explicitly
            from sqlalchemy import or_
            stmt = stmt.where(or_(FlashcardDB.uploaded_to_anki == False, FlashcardDB.uploaded_to_anki.is_(None)))
        except Exception:
            # Fallback: in case SQLAlchemy import fails for some reason, keep going without the filter
            pass
        if where_expr:
            # Very simple filter support: only common fields allowed for safety
            allowed = {"language", "level"}
            try:
                field, value = [s.strip() for s in where_expr.split("=", 1)]
                value = value.strip().strip("'\"")
                if field not in allowed:
                    raise ValueError(f"Filtering by '{field}' is not allowed. Allowed: {sorted(allowed)}")
                if field == "language":
                    stmt = stmt.where(FlashcardDB.language == value)
                elif field == "level":
                    stmt = stmt.where(FlashcardDB.level == value)
            except Exception as e:
                logger.warning(f"Ignoring invalid --where expression '{where_expr}': {e}")
        if limit:
            stmt = stmt.limit(limit)
        cards: List[FlashcardDB] = list(session.exec(stmt))

    if not cards:
        logger.info("No flashcards found to upload.")
        return

    # Prepare notes and media
    note_builds: List[NoteBuild] = [build_note_from_record(c) for c in cards]

    if dry_run:
        logger.info(f"Prepared {len(note_builds)} notes (dry-run). Showing first 3:")
        for nb in note_builds[:3]:
            logger.info(f"Front: {nb.front}\nBack: {nb.back}\nImage: {nb.image_fs_path}\nAudio: {nb.audio_fs_path}")
        return

    # Push to Anki
    anki = AnkiConnect(anki_url)

    # Ensure deck exists
    anki.create_deck(deck_name)

    # Upload media first to avoid missing refs in notes
    for nb in tqdm(note_builds, desc="Uploading media"):
        if nb.image_fs_path and nb.image_filename:
            try:
                anki.store_media_file(filename=nb.image_filename, path=nb.image_fs_path)
            except Exception as e:
                logger.warning(f"Failed to upload image {nb.image_fs_path}: {e}")
        if nb.audio_fs_path and nb.audio_filename:
            try:
                anki.store_media_file(filename=nb.audio_filename, path=nb.audio_fs_path)
            except Exception as e:
                logger.warning(f"Failed to upload audio {nb.audio_fs_path}: {e}")

    # Build addNotes payload
    notes_payload = []
    for nb in note_builds:
        fields = {"Front": nb.front, "Back": nb.back}
        note = {
            "deckName": deck_name,
            "modelName": model_name,
            "fields": fields,
            "options": {"allowDuplicate": True},
            "tags": ["gen-anki-cards"],
        }
        notes_payload.append(note)

    # Add notes
    result = anki.add_notes(notes_payload)
    added = sum(1 for nid in result if isinstance(nid, int)) if isinstance(result, list) else 0
    logger.info(f"Requested adding {len(notes_payload)} notes. Added: {added}")

    # Update DB: mark successfully uploaded cards
    if isinstance(result, list) and result:
        updated = 0
        with Session(engine) as session:
            for card, nid in zip(cards, result):
                if isinstance(nid, int):
                    try:
                        db_obj = session.get(FlashcardDB, card.id)
                        if db_obj:
                            db_obj.uploaded_to_anki = True
                            session.add(db_obj)
                            updated += 1
                    except Exception as e:
                        logger.warning(f"Failed to update upload flag for card id={card.id}: {e}")
            session.commit()
        logger.info(f"Marked {updated} flashcards as uploaded_to_anki=True in DB")


if __name__ == "__main__":
    main()
