"""Durable, private client-visible workflow evidence (standard library only).

Requests commit before a call context yields. Payloads are immutable blobs; an
append-only SQLite ledger serializes concurrent writers and hash-chains events.
This is file integrity evidence, never an assertion of scientific completeness.
"""
from __future__ import annotations

import contextvars
import base64
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from functools import wraps
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import threading
import uuid

active_call_id = contextvars.ContextVar("workflow_trace_active_call_id", default=None)
_poison_lock=threading.Lock()
_poisoned_paths=set()
TERMINALS = frozenset({"completed", "failed", "ended_unresolved"})
_SENSITIVE = frozenset({"authorization", "proxyauthorization", "apikey", "api_key",
    "password", "passwd", "secret", "clientsecret", "access_token", "accesstoken",
    "refresh_token", "refreshtoken", "cookie", "setcookie", "xapikey"})
_AUTH = re.compile(r"(?i)(\b(?:authorization|proxy-authorization|x-api-key|api[_-]?key|password|access[_-]?token)\b[\s\"']*[:=][\s\"']*)(?:bearer\s+|basic\s+)?[^\s\"',;}]+")
_BEARER = re.compile(r"(?i)\b(Bearer|Basic)[ \t]+[A-Za-z0-9._~+/=-]+")


class TraceCaptureError(RuntimeError):
    """Capture failed: callers must abort the action/run, never continue silently."""


def _capture_errors(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except TraceCaptureError:
            if args and isinstance(args[0],TraceStore) and hasattr(args[0],'path'):
                args[0].invalidate()
            raise
        except Exception as error:
            if args and isinstance(args[0],TraceStore) and hasattr(args[0],'path'):
                args[0].invalidate()
            raise TraceCaptureError("workflow trace persistence or validation failed") from error
    return wrapped


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _private(path, mode):
    if os.name == "posix":
        try:
            os.chmod(path, mode)
        except OSError:
            pass  # permissions are best effort; persistence failures are not


def _fsync_dir(path):
    if os.name == "posix":
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


class TraceStore:
    """One run identity per store directory; reopen without run_id to reuse it."""

    @_capture_errors
    def __init__(self, root, run_id=None, secrets=()):
        self.path = Path(root) / "trace" / "workflow"
        self.blobs = self.path / "blobs"
        self.blobs.mkdir(parents=True, exist_ok=True)
        for directory in (self.path.parent, self.path, self.blobs):
            _private(directory, 0o700)
        self.db_path = self.path / "events.sqlite3"
        secrets = tuple(secrets) + tuple(value for key,value in os.environ.items()
            if any(term in key.upper() for term in ('API_KEY','ACCESS_TOKEN','PASSWORD')) and value)
        self.secrets = tuple(sorted({s for s in secrets if isinstance(s, str) and s}, key=len, reverse=True))
        if any(not isinstance(s, str) for s in secrets):
            raise TypeError("secrets must contain strings")
        with self._connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY, run_id TEXT NOT NULL, event TEXT NOT NULL,
                    call_id TEXT, parent_call_id TEXT, stage TEXT NOT NULL,
                    recorded_at TEXT NOT NULL, payload_hash TEXT NOT NULL,
                    payload_size INTEGER NOT NULL, payload_format TEXT NOT NULL,
                    redacted INTEGER NOT NULL, previous_hash TEXT NOT NULL,
                    event_hash TEXT NOT NULL UNIQUE);
                CREATE INDEX IF NOT EXISTS events_call ON events(call_id);
                CREATE TRIGGER IF NOT EXISTS immutable_events_update BEFORE UPDATE ON events
                BEGIN SELECT RAISE(ABORT, 'immutable events'); END;
                CREATE TRIGGER IF NOT EXISTS immutable_events_delete BEFORE DELETE ON events
                BEGIN SELECT RAISE(ABORT, 'immutable events'); END;
                CREATE TRIGGER IF NOT EXISTS immutable_metadata_update BEFORE UPDATE ON metadata
                BEGIN SELECT RAISE(ABORT, 'immutable metadata'); END;
                CREATE TRIGGER IF NOT EXISTS immutable_metadata_delete BEFORE DELETE ON metadata
                BEGIN SELECT RAISE(ABORT, 'immutable metadata'); END;
            """)
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute("SELECT value FROM metadata WHERE key='run_id'").fetchone()
            if existing:
                if run_id is not None and str(run_id) != existing[0]:
                    raise ValueError("store already belongs to a different run_id")
                self.run_id = existing[0]
            else:
                self.run_id = str(run_id) if run_id is not None else uuid.uuid4().hex
                if not self.run_id:
                    raise ValueError("run_id cannot be empty")
                db.execute("INSERT INTO metadata VALUES ('run_id', ?)", (self.run_id,))
            db.commit()
        self._permissions()
        _fsync_dir(self.path)
        _fsync_dir(self.path.parent)

    def invalidate(self):
        """Poison the current process's run scope; swallowed errors cannot retry API."""
        with _poison_lock:
            _poisoned_paths.add(str(self.path.resolve()))
        # Best-effort durable failure flag, even when the failed payload cannot
        # be serialized. Disk-full still propagates and leaves open calls.
        try:
            marker=self.path/('capture-failure.'+uuid.uuid4().hex+'.json')
            with marker.open('xb') as file:
                _private(marker,0o600)
                file.write(_json({'recorded_at':_now(),'reason':'capture persistence or validation failure'}))
                file.flush()
                os.fsync(file.fileno())
            _fsync_dir(self.path)
        except OSError:
            pass

    def assert_writable(self):
        with _poison_lock:
            if str(self.path.resolve()) in _poisoned_paths:
                raise TraceCaptureError('capture already failed in this process; abort and audit before recovery')
        if any(self.path.glob('capture-failure.*.json')):
            raise TraceCaptureError('previous capture failure requires a new run after audit')

    def serialize_payload(self, payload):
        """Auth-filtered strict encoding shared by compatibility mirrors."""
        return self._serialize(payload)

    @contextmanager
    def _connection(self):
        db = sqlite3.connect(self.db_path, timeout=30)
        try:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA synchronous=FULL")
            yield db
        finally:
            db.close()

    def _permissions(self):
        for suffix in ("", "-wal", "-shm"):
            path = Path(str(self.db_path) + suffix)
            if path.exists():
                _private(path, 0o600)

    def sanitize_text(self, text):
        for secret in self.secrets:
            text = text.replace(secret, "[REDACTED]")
        text = _AUTH.sub(lambda m: m.group(1) + "[REDACTED]", text)
        def redact_auth(match):
            if match.group(1).lower() == "basic":
                try:
                    decoded = base64.b64decode(match.group(0).split()[1], validate=True)
                except ValueError:
                    return match.group(0)
                if b":" not in decoded:
                    return match.group(0)
            return match.group(1) + " [REDACTED]"
        return _BEARER.sub(redact_auth, text)

    def _sanitize(self, value):
        if value is None or type(value) in (bool, int):
            return value
        if type(value) is float:
            if not math.isfinite(value):
                raise ValueError("non-finite numbers are not JSON evidence")
            return value
        if type(value) is str:
            return self.sanitize_text(value)
        if type(value) is list:
            return [self._sanitize(item) for item in value]
        if type(value) is dict:
            result = {}
            for key, item in value.items():
                if type(key) is not str:
                    raise TypeError("JSON evidence requires string object keys")
                normalized = re.sub(r"[-\s]", "", key.lower())
                result[self.sanitize_text(key)] = "[REDACTED]" if normalized in _SENSITIVE else self._sanitize(item)
            if len(result) != len(value):
                raise ValueError("redaction would collide object keys")
            return result
        raise TypeError("unsupported evidence type; explicitly convert to JSON, text or bytes")

    def _serialize(self, payload):
        if type(payload) is bytes:
            if payload.lstrip().startswith((b"{", b"[")):
                try:
                    parsed = json.loads(payload)
                except (ValueError, UnicodeError):
                    pass
                else:
                    cleaned = self._sanitize(parsed)
                    if cleaned != parsed:
                        return _json(cleaned), "bytes", True
            # Latin-1 round trips every byte, including non-UTF8 transport bodies.
            cleaned = payload
            for secret in self.secrets:
                cleaned = cleaned.replace(secret.encode("utf-8"), b"[REDACTED]")
            cleaned = self.sanitize_text(cleaned.decode("latin-1")).encode("latin-1")
            return cleaned, "bytes", cleaned != payload
        cleaned = self._sanitize(payload)
        if type(payload) is str:
            return cleaned.encode("utf-8"), "text/utf-8", cleaned != payload
        return _json(cleaned), "json/utf-8", cleaned != payload

    def blob_path(self, digest):
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("invalid blob digest")
        return self.blobs / digest[:2] / digest

    def _write_blob(self, data):
        digest = _sha(data)
        destination = self.blob_path(digest)
        destination.parent.mkdir(exist_ok=True)
        _private(destination.parent, 0o700)
        _fsync_dir(self.blobs)
        temporary = destination.parent / (".tmp-" + uuid.uuid4().hex)
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, destination)
            except FileExistsError:
                if destination.read_bytes() != data:
                    raise OSError("existing content-addressed blob is corrupt")
            _private(destination, 0o600)
            _fsync_dir(destination.parent)
        finally:
            temporary.unlink(missing_ok=True)
        return digest

    @property
    def active_call_id(self):
        return active_call_id.get()

    @_capture_errors
    def emit(self, event, call_id=None, parent_call_id=None, stage=None, payload=None):
        self.assert_writable()
        if not isinstance(event, str) or not event:
            raise ValueError("event must be a nonempty string")
        data, format_, redacted = self._serialize(payload)
        digest = self._write_blob(data)  # cannot begin action if persistence fails
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            if call_id is not None:
                start = db.execute("SELECT * FROM events WHERE call_id=? AND event='started'", (call_id,)).fetchone()
                if event == "started":
                    if start:
                        raise ValueError("call_id already exists")
                    if parent_call_id is not None and not db.execute("SELECT 1 FROM events WHERE call_id=? AND event='started'", (parent_call_id,)).fetchone():
                        raise ValueError("parent call does not exist")
                else:
                    if not start:
                        raise ValueError("call has not started")
                    if parent_call_id is not None and parent_call_id != start["parent_call_id"]:
                        raise ValueError("call parent cannot change")
                    parent_call_id = start["parent_call_id"]
                    if stage is not None and stage != start["stage"]:
                        raise ValueError("call stage cannot change")
                    stage = start["stage"]
                    if db.execute("SELECT 1 FROM events WHERE call_id=? AND event IN ('completed','failed','ended_unresolved')", (call_id,)).fetchone():
                        raise ValueError("call already ended")
            elif event == "started" or event in TERMINALS:
                raise ValueError("call lifecycle event requires call_id")
            if not isinstance(stage, str) or not stage:
                raise ValueError("stage must be a nonempty string")
            previous = db.execute("SELECT id,event_hash FROM events ORDER BY id DESC LIMIT 1").fetchone()
            record = dict(id=previous["id"] + 1 if previous else 1, run_id=self.run_id,
                event=event, call_id=call_id, parent_call_id=parent_call_id, stage=stage,
                recorded_at=_now(), payload_hash=digest, payload_size=len(data),
                payload_format=format_, redacted=int(redacted),
                previous_hash=previous["event_hash"] if previous else "0" * 64)
            record["event_hash"] = _sha(_json(record))
            columns = ",".join(record)
            db.execute(f"INSERT INTO events ({columns}) VALUES ({','.join('?' for _ in record)})", tuple(record.values()))
            db.commit()
        self._permissions()
        return record

    def begin(self, stage, payload, parent_call_id=None):
        call_id = uuid.uuid4().hex
        if parent_call_id is None:
            parent_call_id = active_call_id.get()
        self.emit("started", call_id, parent_call_id, stage, payload)
        return call_id

    start_call = begin

    def record(self, call_id, event, payload):
        return self.emit(event, call_id=call_id, payload=payload)

    def finish(self, call_id, payload=None):
        return self.record(call_id, "completed", payload)

    def fail(self, call_id, error):
        if isinstance(error, BaseException):
            payload = {"error_type": type(error).__name__, "message": self.sanitize_text(str(error))}
        else:
            payload = {"message": self.sanitize_text(str(error))}
        return self.record(call_id, "failed", payload)

    @contextmanager
    def call(self, stage, payload, parent_call_id=None):
        call = TraceCall(self, self.begin(stage, payload, parent_call_id))
        token = active_call_id.set(call.id)
        try:
            yield call
        except BaseException as error:
            if isinstance(error,TraceCaptureError):
                self.invalidate()
                raise
            if not call.ended:
                call.fail(error)
            else:
                # Keep a post-terminal exception as a separate observable event.
                self.emit("post_terminal_exception", parent_call_id=call.id, stage=stage,
                          payload={"call_id": call.id, "error_type": type(error).__name__, "message": self.sanitize_text(str(error))})
            raise
        else:
            if not call.ended:
                call.record("ended_unresolved", {"reason": "context exited without explicit finish"})
        finally:
            active_call_id.reset(token)

    def events(self):
        with self._connection() as db:
            return [dict(row) for row in db.execute("SELECT * FROM events ORDER BY id")]

    read_events = events

    def read_payload(self, record):
        data = self.blob_path(record["payload_hash"]).read_bytes()
        if len(data) != record["payload_size"] or _sha(data) != record["payload_hash"]:
            raise OSError("payload integrity check failed")
        if record["payload_format"] == "bytes":
            return data
        if record["payload_format"] == "text/utf-8":
            return data.decode("utf-8")
        if record["payload_format"] == "json/utf-8":
            return json.loads(data)
        raise ValueError("unknown payload format")

    def _verify(self, rows):
        issues, starts, terminals = [], {}, {}
        previous = "0" * 64
        for expected_id, row in enumerate(rows, 1):
            if row["id"] != expected_id:
                issues.append(f"event sequence gap at {expected_id}")
            if row["run_id"] != self.run_id:
                issues.append(f"mixed run identity at event {row['id']}")
            unsigned = {key: value for key, value in row.items() if key != "event_hash"}
            if row["previous_hash"] != previous or _sha(_json(unsigned)) != row["event_hash"]:
                issues.append(f"event chain/hash mismatch at {row['id']}")
            previous = row["event_hash"]
            try:
                self.read_payload(row)
            except (OSError, ValueError, UnicodeError) as error:
                issues.append(f"blob invalid at event {row['id']}: {type(error).__name__}")
            call_id = row["call_id"]
            if row["parent_call_id"] is not None and row["parent_call_id"] not in starts:
                issues.append(f"missing or future parent at event {row['id']}")
            if row["event"] == "started":
                if call_id in starts or call_id is None:
                    issues.append(f"duplicate or invalid start at event {row['id']}")
                starts[call_id] = row
            elif call_id is not None:
                if call_id not in starts:
                    issues.append(f"event without start at {row['id']}")
                elif (row["parent_call_id"], row["stage"]) != (starts[call_id]["parent_call_id"], starts[call_id]["stage"]):
                    issues.append(f"changed call identity at event {row['id']}")
                if call_id in terminals:
                    issues.append(f"event after terminal at {row['id']}")
                if row["event"] in TERMINALS:
                    terminals.setdefault(call_id, []).append(row["event"])
            elif row["event"] in TERMINALS:
                issues.append(f"terminal without call at {row['id']}")
        completed = failed = 0
        incomplete_ids = []
        for call_id in starts:
            ends = terminals.get(call_id, [])
            if len(ends) > 1:
                issues.append(f"duplicated terminal for call {call_id}")
            if ends == ["completed"]:
                completed += 1
            elif ends == ["failed"]:
                failed += 1
            else:
                incomplete_ids.append(call_id)
        return {"run_id": self.run_id, "total_calls": len(starts), "completed": completed,
            "failed": failed, "incomplete": len(incomplete_ids), "incomplete_call_ids": incomplete_ids,
            "total_events": len(rows), "integrity_ok": not issues, "issues": issues,
            "limitation": "client-visible only", "scientific_completeness": "not assessed"}

    def verify(self):
        result=self._verify(self.events())
        result['capture_failure_markers']=len(list(self.path.glob('capture-failure.*.json')))
        with _poison_lock:
            failed=str(self.path.resolve()) in _poisoned_paths
        result['capture_healthy']=not failed and result['capture_failure_markers']==0
        return result

    @_capture_errors
    def export(self, destination=None):
        """Derived ledger snapshot. Live ledger stays open and unchanged."""
        rows = self.events()
        audit = self._verify(rows)
        audit['capture_failure_markers']=len(list(self.path.glob('capture-failure.*.json')))
        with _poison_lock:
            failed=str(self.path.resolve()) in _poisoned_paths
        audit['capture_healthy']=not failed and audit['capture_failure_markers']==0
        destination = Path(destination) if destination is not None else self.path / "exports" / uuid.uuid4().hex
        destination.mkdir(parents=True, exist_ok=False)
        _private(destination, 0o700)
        ledger = b"".join(_json(row) + b"\n" for row in rows)
        manifest = {"kind": "derived workflow snapshot", "run_id": self.run_id, "recorded_at": _now(),
            "source": str(self.path.resolve()), "ledger_sha256": _sha(ledger),
            "last_event_hash": rows[-1]["event_hash"] if rows else "0" * 64,
            "audit": audit, "blobs": [{"sha256": row["payload_hash"], "size": row["payload_size"],
                "format": row["payload_format"], "source": str(self.blob_path(row["payload_hash"]).resolve())}
                for row in rows]}
        for name, data in (("events.jsonl", ledger), ("manifest.json", _json(manifest))):
            with open(destination / name, "xb") as handle:
                _private(destination / name, 0o600)
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        _fsync_dir(destination)
        return {"manifest": str(destination / "manifest.json"), "ledger": str(destination / "events.jsonl"), "audit": audit}

    seal = export


class TraceCall:
    def __init__(self, store, call_id):
        self.store, self.id, self.call_id, self.ended = store, call_id, call_id, False

    def record(self, event, payload):
        result = self.store.record(self.id, event, payload)
        if event in TERMINALS:
            self.ended = True
        return result

    def finish(self, payload=None):
        return self.record("completed", payload)

    def fail(self, error):
        result = self.store.fail(self.id, error)
        self.ended = True
        return result
