import logging
import os
from datetime import datetime

class _GlogStyleFormatter(logging.Formatter):
    _LEVEL_MAP = {
        logging.INFO: "I",
        logging.WARNING: "W",
        logging.ERROR: "E",
        logging.FATAL: "F",
        logging.DEBUG: "D",
    }

    def format(self, record: logging.LogRecord) -> str:
        level = self._LEVEL_MAP.get(record.levelno, "I")

        now = datetime.fromtimestamp(record.created)
        timestamp = now.strftime("%Y%m%d %H:%M:%S")
        microsecond = f"{now.microsecond:06d}"

        pid = os.getpid()

        filename = record.filename
        lineno = record.lineno

        prefix = (
            f"{level}"
            f"{timestamp}.{microsecond} "
            f"{pid} "
            f"{filename}:{lineno}]"
        )

        output = f"{prefix} {record.getMessage()}"

        # Mirror logging.Formatter: append exception traceback and stack info
        # so logger.exception(...) and exc_info/stack_info kwargs work.
        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            if not output.endswith("\n"):
                output += "\n"
            output += record.exc_text
        if record.stack_info:
            if not output.endswith("\n"):
                output += "\n"
            output += self.formatStack(record.stack_info)

        return output


logger = logging.getLogger("xllm")
logger.setLevel(logging.INFO)
# don't propagate to root logger
logger.propagate = False

if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    # add stream handler if not already added
    _handler = logging.StreamHandler()
    _handler.setFormatter(_GlogStyleFormatter())
    logger.addHandler(_handler)