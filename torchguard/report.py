from dataclasses import dataclass


@dataclass
class SecurityFinding:
    """Represent a security finding."""

    severity: str
    category: str
    message: str