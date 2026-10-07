"""Shared state lifecycle for small calculation objects."""


class Calculation:
    def __init__(self, model):
        self.model = model
        self._reset()

    def _reset(self):
        self.e_tot = None
        self.e_corr = None
        self.converged = False
        self.result = None

    def _finish(self, result, energy, converged, correction=None):
        self.result = result
        self.e_tot = float(energy)
        self.e_corr = None if correction is None else float(correction)
        self.converged = bool(converged)
        return self.e_tot


def require_reference(reference):
    if not reference.converged or reference.result is None:
        raise RuntimeError("kernel() requires a converged reference calculation")
