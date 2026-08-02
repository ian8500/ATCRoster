from __future__ import annotations

import pytest
from cryptography.fernet import Fernet

from atcroster.security.encryption import FieldEncryptionService


def test_current_key_encrypts_with_version_and_decrypts():
    key = Fernet.generate_key().decode()
    service = FieldEncryptionService(f"v2:{key}")

    encrypted = service.encrypt("sensitive-value")

    assert encrypted.startswith("v2.")
    assert service.decrypt(encrypted) == "sensitive-value"
    assert "sensitive-value" not in encrypted


def test_previous_key_decrypts_during_rotation():
    current_key = Fernet.generate_key().decode()
    previous_key = Fernet.generate_key().decode()
    previous_cipher = Fernet(previous_key.encode())
    old_value = "v1." + previous_cipher.encrypt(b"old-secret").decode()
    service = FieldEncryptionService(f"v2:{current_key},v1:{previous_key}")

    assert service.decrypt(old_value) == "old-secret"
    assert service.encrypt("new-secret").startswith("v2.")


def test_legacy_unversioned_ciphertext_tries_each_configured_key():
    current_key = Fernet.generate_key().decode()
    previous_key = Fernet.generate_key().decode()
    legacy_value = Fernet(previous_key.encode()).encrypt(b"legacy-secret").decode()
    service = FieldEncryptionService(f"v2:{current_key},v1:{previous_key}")

    assert service.decrypt(legacy_value) == "legacy-secret"


@pytest.mark.parametrize(
    ("serialized_keys", "message"),
    [
        ("missing-separator", "Invalid field-encryption key version"),
        ("invalid version!:material", "Invalid field-encryption key version"),
        ("v1:not-a-fernet-key", "Invalid field-encryption key material"),
        ("", "Invalid field-encryption key version"),
    ],
)
def test_invalid_key_configuration_is_rejected(serialized_keys, message):
    with pytest.raises(RuntimeError, match=message):
        FieldEncryptionService(serialized_keys)


def test_invalid_ciphertext_and_missing_version_are_rejected():
    key = Fernet.generate_key().decode()
    service = FieldEncryptionService(f"v2:{key}")

    with pytest.raises(ValueError, match="cannot be decrypted"):
        service.decrypt("v2.invalid-ciphertext")
    with pytest.raises(ValueError, match="cannot be decrypted"):
        service.decrypt("v1.invalid-ciphertext")
