class AssemblyError(ValueError):
    pass


class EquivalenceError(AssemblyError):
    pass


class RedundantConnectionError(AssemblyError):
    pass


class RedundantDisconnectionError(AssemblyError):
    pass


class InvalidActionError(AssemblyError):
    pass


class NonexistentAssemblyError(AssemblyError):
    pass


class LabelError(AssemblyError):
    pass
