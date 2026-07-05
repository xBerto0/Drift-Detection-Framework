"""Interfaccia base per i modelli predittivi del framework."""

from abc import ABC, abstractmethod
from typing import Union

import numpy as np


class BaseModel(ABC):
    """Classe astratta che definisce il contratto comune ai modelli predittivi."""

    @abstractmethod
    def predict(self, x) -> Union[float, np.ndarray]:
        """Restituisce la predizione del modello dato l'input x."""

    @abstractmethod
    def get_name(self) -> str:
        """Restituisce il nome identificativo del modello."""
