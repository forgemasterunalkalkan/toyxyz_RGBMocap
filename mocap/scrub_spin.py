"""Number fields scrubbed by dragging sideways, as in 3D tools; a click still types."""
from PySide6.QtCore import Qt, QEvent, Signal
from PySide6.QtWidgets import QDoubleSpinBox


class ScrubSpinBox(QDoubleSpinBox):
    """Drag left/right to change the value (Shift: ten times finer); click without dragging to type.

    valueChanged fires live while dragging, with ``scrubbing`` True. scrubFinished follows on
    release, so callers can preview every step and save once.
    """
    scrubFinished = Signal()
    THRESHOLD = 3

    def __init__(self, per_pixel, parent=None):
        super().__init__(parent)
        self.per_pixel = per_pixel
        self.scrubbing = False
        self._press = None
        self.setButtonSymbols(QDoubleSpinBox.NoButtons)
        self.setKeyboardTracking(False)
        # Qt focuses a click-focus widget before the press reaches any filter, which would make
        # every drag select text. Only Tab focuses; a click without a drag focuses explicitly.
        self.setFocusPolicy(Qt.TabFocus)
        self.lineEdit().setFocusPolicy(Qt.TabFocus)
        self.lineEdit().installEventFilter(self)
        self.lineEdit().setCursor(Qt.SizeHorCursor)
        # Enter ends typing, so the next drag on this field scrubs again.
        self.lineEdit().returnPressed.connect(self.clearFocus)

    # The spin box hands focus events to its line edit directly, past event filters.
    def focusInEvent(self, event):
        super().focusInEvent(event)
        self.lineEdit().setCursor(Qt.IBeamCursor)

    def focusOutEvent(self, event):
        super().focusOutEvent(event)
        self.lineEdit().setCursor(Qt.SizeHorCursor)

    def eventFilter(self, obj, event):
        if obj is not self.lineEdit():
            return super().eventFilter(obj, event)
        kind = event.type()
        if self.hasFocus() and self._press is None:
            pass  # Typing: the line edit keeps its normal selection and cursor behavior.
        elif kind == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            self._press = self._last = event.position().x()
            self._value = self.value()
            return True
        elif kind == QEvent.MouseMove and self._press is not None:
            x = event.position().x()
            if self.scrubbing or abs(x - self._press) >= self.THRESHOLD:
                self.scrubbing = True
                rate = self.per_pixel * (.1 if event.modifiers() & Qt.ShiftModifier else 1)
                # Accumulate unrounded so slow, fine drags still add up between displayed digits.
                self._value += (x - self._last) * rate
                low, high = self.minimum(), self.maximum()
                if self.wrapping():
                    self._value = low + (self._value - low) % (high - low)
                else:
                    self._value = min(high, max(low, self._value))
                self._last = x
                self.setValue(self._value)
            return True
        elif kind == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton and self._press is not None:
            self._press = None
            if self.scrubbing:
                self.scrubbing = False
                self.scrubFinished.emit()
            else:
                self.lineEdit().setFocus(Qt.MouseFocusReason)
                self.selectAll()
            return True
        return super().eventFilter(obj, event)
