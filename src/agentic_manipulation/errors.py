"""Domain-specific exceptions shared across the manipulation pipeline."""


class AgenticManipulationError(RuntimeError):
    """Base exception for recoverable pipeline failures."""


class ConfigurationError(AgenticManipulationError):
    """Raised when runtime configuration is invalid or incomplete."""


class ModelResponseError(AgenticManipulationError):
    """Raised when a model response does not match the requested schema."""


class OllamaUnavailableError(AgenticManipulationError):
    """Raised when the Ollama endpoint cannot serve a request."""


class SemanticValidationError(AgenticManipulationError):
    """Raised when grounded tasks violate command semantics."""


class GraspNetUnavailableError(AgenticManipulationError):
    """Raised when the real GraspNet provider cannot be constructed."""


class PerceptionError(AgenticManipulationError):
    """Raised when calibrated sensor data cannot be produced."""


class ExecutionError(AgenticManipulationError):
    """Raised when a robot motion cannot be executed safely."""
