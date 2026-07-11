"""Root conftest — must be the first thing pytest imports for this test session.

Sets TESTING=true before any `app.*` module is imported anywhere (including by
other conftest.py files or test modules), so `app.config.settings.testing` and
`app.limiter.LIMIT_ASK` / `LIMIT_UPLOAD` resolve to the disabled (10000/minute)
rate limit. Without this, test suites that call /ask or /upload more than a
handful of times (contract tests, integration flow, smoke tests) hit real
production rate limits (10/min ask, 5/min upload) and fail with 429s that have
nothing to do with the behavior under test.

Does not change production rate limit defaults — those still come from
Settings.rate_limit_ask / rate_limit_upload when TESTING is unset.
"""

import os

os.environ.setdefault("TESTING", "true")
