import enum


class OverflowStrategy(str, enum.Enum):
    """
    Enumeration for overflow strategies.
    """

    select_first = "select_first"
    select_random = "select_random"
    select_all = "select_all"
