import os
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Make API key optional to avoid hard dependency in tests/environments
    OPENROUTER_API_KEY: Optional[str] = None

# Load from project root .env if present
settings = Settings(_env_file=os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
