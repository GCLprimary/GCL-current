"""
observer/observer.py

Domain-agnostic waveform observer.

Operates on raw numpy waveform amplitudes — no language-specific logic.
The language consensus path has been moved to the language transducer layer.

Three observers with different band means (Matter/Wave/Data):
  Matter: band_mean=21
  Wave:   band_mean=10
  Data:   band_mean=65
"""
import numpy as np
from typing import Dict, Any, Tuple, Optional
import time
from utils.radial_displacer import radial_displacer


class Observer:
    def __init__(self):
        pass

    def blend(self, data: np.ndarray, band_mean: float = 21.0) -> np.ndarray:
        """
        Normalize and amplify waveform signal.
        Guards against near-constant (saturated) input.
        """
        data_std = float(np.std(data))
        if data_std < 1e-4:
            direction = float(np.mean(data))
            return np.full_like(data, np.clip(direction * 0.5, -1.0, 1.0))

        eq = data - np.mean(data)
        eq = eq / (data_std + 1e-8)

        signal_health = min(1.0, data_std / 0.5)
        boost = (band_mean ** 2) / np.pi * signal_health
        return np.clip(eq * (1 + boost * 0.45), -1.0, 1.0)


class MultiObserver:
    """
    Three-observer waveform consensus — Matter / Wave / Data.

    Operates on raw holographic linkage waveform amplitude.
    Returns a consensus float in [-1, 1] and cumulative perturbation.

    Band means:
      Matter = 21  (physical persistence)
      Wave   = 10  (wave dynamics)
      Data   = 65  (data resolution)
    """

    def __init__(self, num_observers: int = 3):
        self.num_observers       = num_observers
        self.observers           = [Observer() for _ in range(num_observers)]
        self.bands               = [21.0, 10.0, 65.0]
        self.cumulative_perturb  = 0.0
        self.last_consensus_time = time.time()

    def interact(
        self,
        data:        np.ndarray,
        prompt:      str = "",
        iterations:  int = 10,
        prop_result: Dict[str, Any] = None,
    ) -> Tuple[float, float]:
        """
        Compute waveform consensus from three observers.
        Returns (consensus, cumulative_perturbation).
        """
        if len(data) == 0:
            return 0.0, 0.0

        return self._waveform_consensus(data, prompt, iterations, prop_result)

    def _waveform_consensus(
        self,
        data:        np.ndarray,
        prompt:      str,
        iterations:  int,
        prop_result: Optional[Dict[str, Any]],
    ) -> Tuple[float, float]:

        radial_status     = radial_displacer.get_status()
        convergence_boost = max(0.3, radial_status.get("global_clarity", 1.0) * 0.8)

        is_generative = False
        phys_w = wave_w = data_w = 1.0
        if prop_result is not None and prop_result.get("mode") == "generative":
            is_generative = True
            phys_w = prop_result.get("phys_pers", 1.0)
            wave_w = prop_result.get("wave_pers", 1.0)
            data_w = prop_result.get("data_pers", 1.0)
            total  = phys_w + wave_w + data_w + 1e-8
            phys_w = (phys_w / total) * 3
            wave_w = (wave_w / total) * 3
            data_w = (data_w / total) * 3

        perceptions  = []
        role_weights = [phys_w, wave_w, data_w]

        for i, obs in enumerate(self.observers):
            band_mean = self.bands[i % len(self.bands)]
            perc      = obs.blend(data, band_mean=band_mean)
            amp       = float(np.mean(perc))

            if is_generative:
                amp *= role_weights[i % 3] * convergence_boost
            else:
                if i == 0:
                    amp *= 1.8 * convergence_boost
                elif i == 1:
                    amp *= 1.2 * convergence_boost
                else:
                    amp *= 0.8 * convergence_boost

            perceptions.append(amp)

        consensus = np.mean(perceptions)
        for _ in range(iterations):
            props     = [p * (1.05 if i % 2 == 0 else 0.95)
                         for i, p in enumerate(perceptions)]
            consensus = float(np.mean(props))

        final_consensus = float(np.clip(consensus, -1.0, 1.0))
        perturb         = float(np.std(perceptions))
        self.cumulative_perturb = float(np.clip(
            self.cumulative_perturb + perturb * 0.6, -1.0, 1.0
        ))
        self.last_consensus_time = time.time()
        return final_consensus, self.cumulative_perturb

    def get_status(self) -> Dict[str, Any]:
        return {
            "num_observers":       self.num_observers,
            "last_consensus_time": self.last_consensus_time,
            "cumulative_perturb":  round(self.cumulative_perturb, 6),
        }
