
from app.core.config import get_settings

settings=get_settings()

print(f"App Name:{settings.groq_api_key}")