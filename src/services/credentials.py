"""Secure Hugging Face token storage.

On Windows, tokens are stored with the Windows Credential Manager (the same
Credential Locker used by Windows itself). Tokens are never written to the
settings file, logs, or source control.
"""

from __future__ import annotations

import ctypes
import sys
from typing import Protocol

DEFAULT_TARGET = "SimpleHFTrainer/HuggingFace"
CRED_TYPE_GENERIC = 1
CRED_PERSIST_LOCAL_MACHINE = 2
ERROR_NOT_FOUND = 1168


class CredentialStoreError(RuntimeError):
    """Raised when the credential store cannot fulfil a request."""


class CredentialStore(Protocol):
    @property
    def available(self) -> bool: ...

    def get_token(self, target: str = DEFAULT_TARGET) -> str | None: ...

    def set_token(self, token: str, target: str = DEFAULT_TARGET) -> None: ...

    def delete_token(self, target: str = DEFAULT_TARGET) -> None: ...


class NullCredentialStore:
    """Used when no secure store is available (for example on Linux CI)."""

    available = False

    def get_token(self, target: str = DEFAULT_TARGET) -> str | None:
        return None

    def set_token(self, token: str, target: str = DEFAULT_TARGET) -> None:
        raise CredentialStoreError("No secure credential store is available on this platform.")

    def delete_token(self, target: str = DEFAULT_TARGET) -> None:
        return None


if sys.platform == "win32":  # pragma: no cover - exercised by Windows-only tests
    from ctypes import wintypes

    class _FILETIME(ctypes.Structure):
        _fields_ = [
            ("dwLowDateTime", wintypes.DWORD),
            ("dwHighDateTime", wintypes.DWORD),
        ]

    class _CREDENTIALW(ctypes.Structure):
        _fields_ = [
            ("Flags", wintypes.DWORD),
            ("Type", wintypes.DWORD),
            ("TargetName", wintypes.LPWSTR),
            ("Comment", wintypes.LPWSTR),
            ("LastWritten", _FILETIME),
            ("CredentialBlobSize", wintypes.DWORD),
            ("CredentialBlob", ctypes.POINTER(ctypes.c_byte)),
            ("Persist", wintypes.DWORD),
            ("AttributeCount", wintypes.DWORD),
            ("Attributes", ctypes.c_void_p),
            ("TargetAlias", wintypes.LPWSTR),
            ("UserName", wintypes.LPWSTR),
        ]

    class WindowsCredentialStore:
        """Thin wrapper over ``CredReadW`` / ``CredWriteW`` / ``CredDeleteW``."""

        available = True

        def __init__(self) -> None:
            self._advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
            self._advapi32.CredReadW.argtypes = [
                wintypes.LPCWSTR,
                wintypes.DWORD,
                wintypes.DWORD,
                ctypes.POINTER(ctypes.POINTER(_CREDENTIALW)),
            ]
            self._advapi32.CredReadW.restype = wintypes.BOOL
            self._advapi32.CredWriteW.argtypes = [
                ctypes.POINTER(_CREDENTIALW),
                wintypes.DWORD,
            ]
            self._advapi32.CredWriteW.restype = wintypes.BOOL
            self._advapi32.CredDeleteW.argtypes = [
                wintypes.LPCWSTR,
                wintypes.DWORD,
                wintypes.DWORD,
            ]
            self._advapi32.CredDeleteW.restype = wintypes.BOOL
            self._advapi32.CredFree.argtypes = [ctypes.c_void_p]
            self._advapi32.CredFree.restype = None

        def get_token(self, target: str = DEFAULT_TARGET) -> str | None:
            pointer = ctypes.POINTER(_CREDENTIALW)()
            ok = self._advapi32.CredReadW(target, CRED_TYPE_GENERIC, 0, ctypes.byref(pointer))
            if not ok:
                error = ctypes.get_last_error()
                if error == ERROR_NOT_FOUND:
                    return None
                raise CredentialStoreError(
                    f"Windows Credential Manager read failed (error {error})."
                )
            try:
                credential = pointer.contents
                size = int(credential.CredentialBlobSize)
                if size == 0:
                    return None
                raw = ctypes.string_at(credential.CredentialBlob, size)
                return raw.decode("utf-16-le")
            finally:
                self._advapi32.CredFree(pointer)

        def set_token(self, token: str, target: str = DEFAULT_TARGET) -> None:
            token = token.strip()
            if not token:
                raise CredentialStoreError("Refusing to store an empty token.")
            blob = token.encode("utf-16-le")
            buffer = ctypes.create_string_buffer(blob, len(blob))
            credential = _CREDENTIALW()
            credential.Type = CRED_TYPE_GENERIC
            credential.TargetName = target
            credential.Comment = "Simple HF Trainer Hugging Face token"
            credential.CredentialBlobSize = len(blob)
            credential.CredentialBlob = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_byte))
            credential.Persist = CRED_PERSIST_LOCAL_MACHINE
            credential.UserName = "huggingface"
            if not self._advapi32.CredWriteW(ctypes.byref(credential), 0):
                raise CredentialStoreError(
                    f"Windows Credential Manager write failed (error {ctypes.get_last_error()})."
                )

        def delete_token(self, target: str = DEFAULT_TARGET) -> None:
            if not self._advapi32.CredDeleteW(target, CRED_TYPE_GENERIC, 0):
                error = ctypes.get_last_error()
                if error != ERROR_NOT_FOUND:
                    raise CredentialStoreError(
                        f"Windows Credential Manager delete failed (error {error})."
                    )


def create_credential_store() -> CredentialStore:
    """Return the best secure store available on this platform."""
    if sys.platform == "win32":
        try:
            return WindowsCredentialStore()
        except OSError:
            return NullCredentialStore()
    return NullCredentialStore()
