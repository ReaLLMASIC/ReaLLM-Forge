#!/usr/bin/env python3
"""
hf_auth.py

Run once before training/inference:

    python hf_auth.py

It checks for an existing Hugging Face token. If none is found,
it securely asks for one and saves it using huggingface_hub.login(),
the same local auth/cache mechanism used by Transformers/Datasets.
"""

import getpass
import sys

from huggingface_hub import get_token, login, whoami


def ensure_hf_token() -> str:
    """
    Ensure a Hugging Face token is available locally.

    Returns:
        str: The active Hugging Face token.
    """
    token = get_token()

    if token:
        print("✅ Hugging Face token found. Already authenticated.")

        # Optional sanity check: verify the token works.
        try:
            user = whoami(token=token)
            username = user.get("name") or user.get("fullname") or "unknown user"
            print(f"👤 Authenticated as: {username}")
        except Exception as exc:
            print("⚠️ A token was found, but it could not be verified.")
            print(f"Reason: {exc}")
            print("You may need to paste a fresh token.")

            token = None

    if token is None:
        print("⚠️ Hugging Face token not found.")
        print("Paste a Hugging Face access token. Input will be hidden.")

        hf_token = getpass.getpass("HF token: ").strip()

        if not hf_token:
            print("❌ No token entered. Exiting.")
            sys.exit(1)

        try:
            # This saves the token in the normal Hugging Face local cache.
            # Default location is usually ~/.cache/huggingface/token,
            # unless HF_HOME is set.
            login(token=hf_token, add_to_git_credential=False)

            print("✅ Token saved locally.")

            user = whoami(token=hf_token)
            username = user.get("name") or user.get("fullname") or "unknown user"
            print(f"👤 Authenticated as: {username}")

            token = hf_token

        except Exception as exc:
            print("❌ Failed to authenticate with the provided token.")
            print(f"Reason: {exc}")
            sys.exit(1)

    return token


if __name__ == "__main__":
    ensure_hf_token()
