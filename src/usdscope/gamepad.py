"""A game controller read each frame as plain data (`Pad.poll`): its sticks, triggers and
buttons, the sticks' dead middle taken out, which buttons went down since the last read.
Plugged in or pulled out at any time; no controller, no state (None).

    pad = Pad()
    state = pad.poll()            # PadState, or None with no controller
    if state and state.pressed("x"): ...
    state.left                    # (x, y): right and up are positive, -1 to 1

Read through pyglet's controllers. On Linux (evdev: the user must be able to read
/dev/input/event*) the device is polled here each frame, so nothing needs pyglet's own event
loop to run; on Windows an XInput pad is read by pyglet's own thread and a DirectInput one by
pyglet's event loop, which Newton's viewer steps each frame. A `PadState` can be made by hand
(tests, an agent driving a window).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

#: Buttons, as pyglet's controllers name them (an Xbox pad's A, B, X, Y, the shoulders LB and
#: RB, Start, Back (View), Guide, the sticks pressed in).
BUTTONS = (
    "a", "b", "x", "y", "leftshoulder", "rightshoulder", "start", "back", "guide",
    "leftstick", "rightstick",
)  # fmt: skip
#: Controllers whose X and Y come crossed (by name): Linux's xpad driver sends an Xbox pad's X
#: and Y as BTN_X and BTN_Y, while pyglet, with no mapping for the pad, reads them by the Linux
#: gamepad rules (BTN_NORTH, the same code as BTN_X, is the top button: Y).
CROSSED_XY = ("Xbox", "Microsoft X-Box")
#: Of a stick's throw, how much in the middle reads as nothing (its rest is never exactly 0).
DEAD = 0.12
#: Seconds between looks for a controller when there is none.
LOOK = 1.0


@dataclass(frozen=True)
class PadState:
    """A controller at one moment: each stick's (x, y) (right and up positive, the dead middle
    taken out, at most 1 long), each trigger 0 to 1, the d-pad's (x, y) in -1, 0, 1, the
    buttons held, and those that went down since the last read (`down`)."""

    left: tuple[float, float] = (0.0, 0.0)
    right: tuple[float, float] = (0.0, 0.0)
    left_trigger: float = 0.0
    right_trigger: float = 0.0
    dpad: tuple[int, int] = (0, 0)
    held: frozenset[str] = field(default_factory=frozenset)
    down: frozenset[str] = field(default_factory=frozenset)
    name: str = ""

    def holds(self, button: str) -> bool:
        return button in self.held

    def pressed(self, button: str) -> bool:
        """Whether `button` went down since the last read."""
        return button in self.down


def dead_zoned(x: float, y: float, dead: float = DEAD) -> tuple[float, float]:
    """A stick's (x, y) with its middle `dead` of throw read as nothing and the rest stretched
    to fill 0 to 1 (radially: the way it points kept), at most 1 long."""
    size = math.hypot(x, y)
    if size <= dead:
        return 0.0, 0.0
    scaled = min((size - dead) / (1.0 - dead), 1.0)
    return x / size * scaled, y / size * scaled


class Pad:
    """The first game controller there is, read each frame (`poll`)."""

    def __init__(self, dead: float = DEAD) -> None:
        self.dead = dead
        self.controller: Any = None
        self.looked = -math.inf
        self.was: frozenset[str] = frozenset()

    def _find(self) -> None:
        now = time.monotonic()
        if now - self.looked < LOOK:
            return
        self.looked = now
        try:
            import warnings

            import pyglet

            with warnings.catch_warnings():  # pads without SDL's mapping: Linux's own is used
                warnings.simplefilter("ignore")
                found = pyglet.input.get_controllers()
            if found:
                found[0].open()
                self.controller = found[0]
        except Exception:  # no input devices to read here (no permission, no backend)
            self.controller = None

    def poll(self) -> PadState | None:
        """The controller as it is now, or None when there is none."""
        if self.controller is None:
            self._find()
            if self.controller is None:
                return None
        c = self.controller
        device = getattr(c, "device", None)
        try:
            if device is not None and hasattr(device, "poll") and hasattr(device, "select"):
                while device.poll():  # evdev: what it has sent since the last frame
                    device.select()
                if not getattr(device, "connected", True):
                    raise OSError("pulled out")
        except OSError:
            self.controller, self.was = None, frozenset()
            return None
        held = frozenset(b for b in BUTTONS if getattr(c, b, False))
        if any(n in str(getattr(c, "name", "")) for n in CROSSED_XY):
            held = frozenset({"x": "y", "y": "x"}.get(b, b) for b in held)
        down, self.was = held - self.was, held
        # pyglet's raw y axes read a stick pushed up as negative (the HID way: evdev, DirectInput,
        # macOS), except its XInputController's, which reads up as positive as XInput sends it.
        # Up is positive here.
        up = 1.0 if type(c).__name__ == "XInputController" else -1.0
        return PadState(
            left=dead_zoned(float(c.leftx), up * float(c.lefty), self.dead),
            right=dead_zoned(float(c.rightx), up * float(c.righty), self.dead),
            left_trigger=max(0.0, float(c.lefttrigger)),
            right_trigger=max(0.0, float(c.righttrigger)),
            dpad=(round(float(c.dpadx)), round(float(c.dpady))),
            held=held,
            down=down,
            name=str(getattr(c, "name", "")),
        )
