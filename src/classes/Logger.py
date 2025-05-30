import logging
from logging.handlers import TimedRotatingFileHandler

file_handler = TimedRotatingFileHandler(
    filename="bot.log",
    when="midnight",  # Ротация каждый день в полночь
    interval=1,
    backupCount=30,  # Хранить логи за последние 30 дней
)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

# Logger define
logger = logging.getLogger("my_bot_logger")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)