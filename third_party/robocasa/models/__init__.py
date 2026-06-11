import os

assets_root = os.environ.get(
    "REVAMP_ROBOCASA_ASSETS_ROOT",
    os.path.join(os.path.dirname(__file__), "assets"),
)
