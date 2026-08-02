"""Versioned encryption for security-sensitive application fields."""

from __future__ import annotations

import re

from cryptography.fernet import Fernet, InvalidToken


class FieldEncryptionService:
    """Encrypt with the current key and decrypt with the matching key version."""

    def __init__(self, serialized_keys: str) -> None:
        self._ciphers = self._parse_keys(serialized_keys)

    @staticmethod
    def _parse_keys(serialized_keys: str) -> tuple[tuple[str, Fernet], ...]:
        result: list[tuple[str, Fernet]] = []
        for item in serialized_keys.split(","):
            version, separator, key = item.strip().partition(":")
            if not separator or not re.fullmatch(r"[A-Za-z0-9_-]{1,20}", version):
                raise RuntimeError("Invalid field-encryption key version.")
            try:
                result.append((version, Fernet(key.encode())))
            except (ValueError, TypeError) as exc:
                raise RuntimeError("Invalid field-encryption key material.") from exc
        if not result:
            raise RuntimeError("At least one field-encryption key is required.")
        return tuple(result)

    def ciphers(self) -> list[tuple[str, Fernet]]:
        """Return the parsed key ring for the temporary compatibility API."""
        return list(self._ciphers)

    def encrypt(self, value: str) -> str:
        version, cipher = self._ciphers[0]
        return f"{version}.{cipher.encrypt(value.encode()).decode()}"

    def decrypt(self, value: str) -> str:
        version, separator, ciphertext = value.partition(".")
        if separator:
            candidates = (
                cipher
                for candidate_version, cipher in self._ciphers
                if candidate_version == version
            )
        else:
            ciphertext = value
            candidates = (cipher for _version, cipher in self._ciphers)
        for cipher in candidates:
            try:
                return cipher.decrypt(ciphertext.encode()).decode()
            except InvalidToken:
                continue
        raise ValueError("Encrypted field cannot be decrypted with configured keys.")
