"""应用入口（FastAPI + uvicorn）"""
import logging
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import config


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in ("httpx", "httpcore", "urllib3", "openai._base_client"):
        logging.getLogger(name).setLevel(logging.WARNING)


def main():
    setup_logging()
    logger = logging.getLogger("app")

    for w in config.validate():
        logger.warning(w)

    logger.info(f"ASR API: {config.ASR_API_BASE} (model={config.ASR_MODEL})")
    logger.info(f"LLM API: {config.LLM_API_BASE} (model={config.LLM_MODEL})")
    logger.info(f"启动服务: http://{config.GRADIO_HOST}:{config.GRADIO_PORT}")

    import uvicorn

    uvicorn.run(
        "server:app",
        host=config.GRADIO_HOST,
        port=config.GRADIO_PORT,
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    main()
