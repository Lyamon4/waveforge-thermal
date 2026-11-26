"""Exclusive resumable evidence journal. No overwrite and no cached failed jobs."""

import json
import os
import tempfile
from pathlib import Path

from .contracts import content_hash


def _exclusive_json(path: Path, payload: dict) -> None:
    data = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False).encode(
        "utf-8"
    )
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".partial", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        # Atomic exclusive publication; unlike replace, cannot erase prior data.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class RunJournal:
    def __init__(self, folder: Path, manifest: dict):
        self.folder = Path(folder)
        self.manifest = json.loads(json.dumps(manifest, allow_nan=False))
        self._handle = None

    def __enter__(self):
        self.folder.mkdir(parents=True, exist_ok=True)
        handle = (self.folder / ".lock").open("a+b")
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise RuntimeError("another runner is active in this journal") from exc
        self._handle = handle
        try:
            path = self.folder / "manifest.json"
            if path.exists():
                if json.loads(path.read_text(encoding="utf-8")) != self.manifest:
                    raise ValueError(
                        "resume manifest differs from frozen source/protocol"
                    )
            else:
                _exclusive_json(path, self.manifest)
            (self.folder / "jobs").mkdir(exist_ok=True)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_):
        if self._handle is not None:
            self._handle.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            finally:
                self._handle.close()
                self._handle = None

    def obtain(self, key: dict, calculate) -> dict:
        if self._handle is None:
            raise RuntimeError("journal requires an active exclusive lease")
        path = self.folder / "jobs" / f"{content_hash(key)}.json"
        if path.exists():
            envelope = json.loads(path.read_text(encoding="utf-8"))
            body = {"key": envelope["key"], "result": envelope["result"]}
            if envelope["key"] != key or content_hash(body) != envelope["sha256"]:
                raise ValueError("cached evidence hash mismatch")
            return envelope["result"]
        result = calculate()
        body = {"key": key, "result": result}
        _exclusive_json(path, dict(body, sha256=content_hash(body)))
        return result
