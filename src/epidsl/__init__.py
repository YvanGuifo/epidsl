"""EpiDSL: a domain-specific language for compositional epidemiological modelling."""

from epidsl.checks import Diagnostic, Severity, check
from epidsl.errors import CompositionError, EpiError, EpiSyntaxError
from epidsl.ir import EpiModel, SimulationConfig
from epidsl.parser import parse

__all__ = [
    "CompositionError",
    "Diagnostic",
    "EpiError",
    "EpiModel",
    "EpiSyntaxError",
    "Severity",
    "SimulationConfig",
    "check",
    "parse",
]
