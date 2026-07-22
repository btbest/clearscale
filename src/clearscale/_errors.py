class NoSuchCoordinateSystemError(ValueError):
    """
    Raised by Multiscale.as_ref when called with a name that does not match any of the Multiscale's coordinate systems.
    """

    def __init__(self, name):
        super().__init__(f"This Multiscale does not specify a coordinate system named: {name!r}")


class MismatchingMultiscaleError(ValueError):
    """
    Raised by Transform.with_resolved when a Multiscale provided for `path`
    does not have a coordinate system named `name` as expected.
    """

    path: str
    name: str

    def __init__(self, path: str, name: str):
        super().__init__(f"No coordinate system named {name!r} in Multiscale provided for path {path!r}")
        self.path = path
        self.name = name
