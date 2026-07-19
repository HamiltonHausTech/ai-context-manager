"""Typed persistence failures shared by memory backends."""


class StorageError(RuntimeError):
    """Base class for persistence failures."""


class StorageReadError(StorageError):
    """A memory backend could not read or decode its data."""


class StorageWriteError(StorageError):
    """A memory backend could not durably write a requested change."""
