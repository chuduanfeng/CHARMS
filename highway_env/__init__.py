import os
import sys

from gymnasium.envs.registration import register


__version__ = "1.10.1"

try:
    from farama_notifications import notifications

    if "highway_env" in notifications and __version__ in notifications["gymnasium"]:
        print(notifications["highway_env"][__version__], file=sys.stderr)

except Exception:  # nosec
    pass

# Hide pygame support prompt
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"


def _register_highway_envs():
    """Import the envs module so that envs register themselves."""

    from highway_env.envs.common.abstract import MultiAgentWrapper

    # level1_env.py
    register(
        id="level1-v0",
        entry_point="highway_env.envs.level1_env:Level1Env",
    )

    # level2_env.py
    register(
        id="level2-v0",
        entry_point="highway_env.envs.level2_env:Level2Env",
    )

    # test_env.py
    register(
        id="test1-v0",
        entry_point="highway_env.envs.test_env:Test1Env",
    )

    register(
        id="test2-v0",
        entry_point="highway_env.envs.test_env:Test2Env",
    )

    # llm-director
    register(
        id="director-v0",
        entry_point="highway_env.envs.director_env:DirectorEnv",
    )


_register_highway_envs()
