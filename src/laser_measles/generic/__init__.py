__version__ = "0.0.0"

from .components.process_births import BirthsParams, BirthsProcess
from .components.process_births import BirthsConstantPopProcess
from .core import compute
from .components.process_importation import ImportationParams, InfectRandomAgentsProcess
from .components.process_infection import InfectionParams, InfectionProcess
from .components.process_exposure import ExposureParams, ExposureProcess
from .model import Model
from .components.process_susceptibility import SusceptibilityParams, SusceptibilityProcess
from .components.process_transmission import TransmissionParams, TransmissionProcess

# Backward compatibility aliases
Births = BirthsProcess
BirthsConstantPop = BirthsConstantPopProcess
Exposure = ExposureProcess
Infection = InfectionProcess
Susceptibility = SusceptibilityProcess
Transmission = TransmissionProcess
InfectRandomAgents = InfectRandomAgentsProcess
from .params import (
    SimulationParams,
    BirthParams,
    ExposureParams,
    InfectionParams,
    ImportationParams,
    TransmissionParams,
    GenericModelParams,
)

__all__ = [
    # Process classes
    "BirthsProcess",
    "BirthsConstantPopProcess", 
    "InfectRandomAgentsProcess",
    "InfectionProcess",
    "ExposureProcess",
    "SusceptibilityProcess",
    "TransmissionProcess",
    # Parameter classes
    "BirthsParams",
    "ExposureParams",
    "InfectionParams",
    "ImportationParams",
    "SusceptibilityParams", 
    "TransmissionParams",
    # Backward compatibility aliases
    "Births",
    "BirthsConstantPop",
    "Exposure", 
    "Infection",
    "Susceptibility",
    "Transmission",
    "InfectRandomAgents",
    # Other exports
    "Model",
    "compute",
    "SimulationParams",
    "BirthParams",
    "GenericModelParams",
]
