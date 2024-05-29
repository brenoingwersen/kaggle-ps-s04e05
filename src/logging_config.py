import logging
import logging.config


def setup_logging():
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "level": "INFO",
            },
            "file": {
                "class": "logging.FileHandler",
                "filename": "project.log",
                "formatter": "standard",
                "level": "INFO",
            },
        },
        "root": {
            "handlers": ["console", "file"],
            "level": "INFO",
        },
    })


# Call setup_logging() when this module is imported
setup_logging()
