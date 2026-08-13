"""
Optional cursor control.

pyautogui is imported lazily. On a headless server (Streamlit Cloud, Docker,
CI) importing it at module load raises and takes the whole app down, so the
import is deferred and every failure degrades to "unavailable" instead of a
crash.
"""

from __future__ import annotations

import logging
import platform

from .config import CLASS_LEFT, CLASS_REST, CLASS_RIGHT

logger = logging.getLogger(__name__)

MOVE_DISTANCE = 200      # pixels per command
MOVE_DURATION = 0.25     # seconds of easing


class MouseController:
    def __init__(self) -> None:
        self._pyautogui = None
        self.available = False
        self.reason = "Not initialised"

    def initialise(self) -> tuple[bool, str]:
        try:
            import pyautogui  # noqa: PLC0415 - deliberately lazy
            pyautogui.FAILSAFE = True
            pyautogui.size()
            self._pyautogui = pyautogui
            self.available = True
            self.reason = "Cursor control ready"
        except Exception as exc:                       # noqa: BLE001
            self.available = False
            self.reason = f"Cursor control unavailable: {exc}. {self.permission_hint()}"
            logger.warning(self.reason)
        return self.available, self.reason

    @staticmethod
    def permission_hint() -> str:
        system = platform.system()
        if system == "Darwin":
            return ("On macOS grant Accessibility rights: System Settings > Privacy & "
                    "Security > Accessibility, add your terminal or IDE, then restart it.")
        if system == "Windows":
            return "On Windows run the terminal as Administrator."
        if system == "Linux":
            return "On Linux an X11/Wayland session with python3-xlib is required."
        return "A local desktop session is required."

    def move(self, class_index: int) -> str | None:
        """Move the cursor for one decoded command. Returns a description."""
        if not self.available or self._pyautogui is None:
            return None
        try:
            pag = self._pyautogui
            width, height = pag.size()
            x, y = pag.position()

            if class_index == CLASS_LEFT:
                target = (max(50, x - MOVE_DISTANCE), y)
                label = "LEFT"
            elif class_index == CLASS_RIGHT:
                target = (min(width - 50, x + MOVE_DISTANCE), y)
                label = "RIGHT"
            elif class_index == CLASS_REST:
                target = (width // 2, height // 2)
                label = "CENTER"
            else:
                return None

            pag.moveTo(target[0], target[1], duration=MOVE_DURATION)
            return f"{label} -> {target}"
        except Exception as exc:                        # noqa: BLE001
            logger.error("Cursor move failed: %s", exc)
            self.available = False
            self.reason = f"Cursor control stopped: {exc}. {self.permission_hint()}"
            return None
