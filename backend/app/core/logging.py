import logging
import re

from backend.app.core.config import Settings

# Values matching these patterns are redacted from log records.
_SECRET_PATTERNS = (
    re.compile(r"(password|secret|api[_-]?key|token|authorization)\s*[:=]\s*\S+", re.IGNORECASE),
)


class _SecretRedactionFilter(logging.Filter):
    """Redact anything that looks like a secret from log output."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = _SECRET_PATTERNS.sub(r"\1=<redacted>", record.msg)
        if record.args:
            record.args = tuple(
                _SECRET_PATTERNS.sub(r"\1=<redacted>", str(arg)) if isinstance(arg, str) else arg
                for arg in record.args
            )
        return True


def configure_logging(settings: Settings) -> None:
    """Configure root logging. Idempotent: skips if handlers already attached."""
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(settings.log_level.upper())
        return
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    root.addFilter(_SecretRedactionFilter())

