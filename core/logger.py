import logging
import sys
from colorama import init as colorama_init, Fore, Style

colorama_init()                       # инициализация (делает Windows ANSI-friendly)

_FMT  = "[%(name)s] %(levelname)s %(asctime)s - %(message)s"
_DATE = "%d-%m-%Y %H:%M:%S"

_LEVEL_COLORS = {
    logging.DEBUG:    Fore.CYAN,      # DEBUG  → голубой
    logging.INFO:     Fore.WHITE,     # INFO   → белый
    logging.WARNING:  Fore.YELLOW,    # WARNING→ жёлтый
    logging.ERROR:    Fore.RED,       # ERROR  → красный
    logging.CRITICAL: Fore.RED + Style.BRIGHT,  # CRITICAL — ярко-красный
}

class ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        color = _LEVEL_COLORS.get(record.levelno, "")
        reset = Style.RESET_ALL
        record.levelname = f"{color}{record.levelname}{reset}"
        return super().format(record)

def setup_logger(name: str, worker_id: int | None = None) -> logging.Logger:
    """Настройка логгера с цветами по уровню"""
    logger_name = f"{name}-{worker_id}" if worker_id is not None else name
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False          # чтобы не дублировалось в root-логгер

    if not logger.handlers:  # ← ключ: добавляем ОДИН раз
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(ColorFormatter(_FMT, datefmt=_DATE))
        logger.addHandler(handler)
    else:
        # хэндлер уже есть → можно обновить уровень/формат по-желанию
        logger.handlers[0].setFormatter(logging.Formatter(_FMT, datefmt=_DATE))

    return logger