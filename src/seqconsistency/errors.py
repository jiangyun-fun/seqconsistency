"""Error hierarchy for seqconsistency.

All errors in this module are execution/input/configuration failures: they abort the run
with a nonzero exit code (2). Scientific problems (distribution shifts, metadata
conflicts) are *findings*, not exceptions — see :mod:`seqconsistency.models`.
"""

from __future__ import annotations


class SeqConsistencyError(Exception):
    """Base class for all execution-level seqconsistency errors (exit code 2)."""

    def __init__(self, message: str, *, context: dict[str, str] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict[str, str] = dict(context) if context else {}


class ConfigError(SeqConsistencyError):
    """Invalid CLI arguments, policy file, or resource settings."""


class ManifestError(SeqConsistencyError):
    """Invalid, incomplete, or contradictory input manifest."""


class PathError(SeqConsistencyError):
    """Output path validation failure (collision, missing parent, permission)."""


class InputDataError(SeqConsistencyError):
    """Structural failure in input data (malformed FASTQ/BAM, broken pairing).

    ``source`` names the file being scanned and ``location`` a record/offset
    identifier when known, so users can repair the input.
    """

    def __init__(
        self,
        message: str,
        *,
        source: str | None = None,
        location: str | None = None,
        context: dict[str, str] | None = None,
    ) -> None:
        full = message
        if source is not None:
            full = f"{source}: {message}"
        if location is not None:
            full = f"{full} (at {location})"
        super().__init__(full, context=context)
        self.source = source
        self.location = location


class DependencyError(SeqConsistencyError):
    """A requested analysis needs a dependency that is unavailable."""


class OutputError(SeqConsistencyError):
    """Failure while writing or publishing persistent outputs."""
