"""EpiDSL: a domain-specific language for compositional epidemiological modelling."""

from epidsl.checks import Diagnostic, Severity, check
from epidsl.ir import EpiModel, SimulationConfig
from epidsl.parser import EpiSyntaxError, parse

__all__ = [
    "Diagnostic",
    "EpiModel",
    "EpiSyntaxError",
    "Severity",
    "SimulationConfig",
    "check",
    "parse",
]
