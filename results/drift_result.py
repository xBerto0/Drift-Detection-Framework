"""Risultato prodotto da un drift detector."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class DriftResult:
    """Rappresenta l'esito di un controllo di drift effettuato da un detector."""

    detector_name: str
    drift_detected: bool
    drift_type: str
    score: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict) # ci dice in quali feature è stato rilevato il drift
