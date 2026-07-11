"""app/limiter.py — slowapi Limiter singleton shared across routes.

Rate limits are read from Settings so they can be overridden via environment
variables without changing code:

    RATE_LIMIT_ASK=20/minute      # default: 10/minute
    RATE_LIMIT_UPLOAD=10/minute   # default: 5/minute

When TESTING=true the limits are set to 10000/minute, effectively disabling
rate limiting in the test suite without skipping the middleware path entirely.

Usage in route files:
    from app.limiter import limiter, LIMIT_ASK, LIMIT_UPLOAD
    @limiter.limit(LIMIT_ASK)
    def my_route(request: Request, ...):  # Request must be a parameter
        ...
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import settings

# slowapi's Limiter auto-reads a ".env" in the cwd via starlette.Config for its
# own RATELIMIT_* overrides. We don't use that mechanism (limits come from our
# own app.config.settings instead), and on Windows reading the repo's UTF-8
# .env with the default cp1252 codec raises UnicodeDecodeError. Point it at a
# file that doesn't exist so slowapi skips the read entirely (starlette.Config
# only parses the file if os.path.isfile() is true).
limiter = Limiter(key_func=get_remote_address, config_filename="__no_slowapi_env__")

# Resolved once at import time; settings already loaded from .env / env vars.
LIMIT_ASK: str = "10000/minute" if settings.testing else settings.rate_limit_ask
LIMIT_UPLOAD: str = "10000/minute" if settings.testing else settings.rate_limit_upload
