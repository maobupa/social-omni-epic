"""social_omni_epic package.

IMPORTANT: Sotopia selects its storage backend at IMPORT TIME based on the
SOTOPIA_STORAGE_BACKEND environment variable. With the default ("redis"),
even *constructing* an AgentProfile/EnvironmentProfile opens a Redis
connection. We never use Redis — we pass profiles in memory — so we force
the local backend here, before any `import sotopia` can run.

This runs whenever any `social_omni_epic.*` submodule is imported, which
always happens before our sotopia-touching modules (episode_runner,
sotopia_bridge) load their `from sotopia ...` imports.
"""
import os

os.environ.setdefault("SOTOPIA_STORAGE_BACKEND", "local")
