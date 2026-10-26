import logging
import os

# VoxCPM and funasr draw tqdm bars per sentence; in a server they are just log noise.
os.environ.setdefault("TQDM_DISABLE", "1")


def main() -> None:
    import faulthandler
    import signal

    import uvicorn

    # `kill -USR1 <pid>` prints every thread's stack to the log: the way to see what a stuck TTS
    # instance is doing (py-spy needs root on macOS).
    faulthandler.register(signal.SIGUSR1, all_threads=True)

    from .config import settings

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run("vgworker.app:app", host=settings.host, port=settings.port, ws_max_size=16 * 1024 * 1024)


if __name__ == "__main__":
    main()
