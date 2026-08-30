from __future__ import annotations


class AdapterError(RuntimeError):
    """Base error for external memory repository failures."""


class AdapterUnavailableError(AdapterError):
    pass


class AdapterAuthenticationError(AdapterError):
    pass


class AdapterConfigurationError(AdapterError):
    pass


class AdapterWriteError(AdapterError):
    pass


class AdapterSearchError(AdapterError):
    pass


class AdapterProtocolError(AdapterError):
    pass
