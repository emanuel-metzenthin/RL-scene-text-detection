from ray.rllib.utils.filter import Filter


class NormalizeFilter(Filter):
    """
    Filter that normalizes
    """

    def copy(self):
        return NormalizeFilter()

    def as_serializable(self):
        return self.copy()

    def __call__(self, x, update=True):
        return x / 255

    def __repr__(self):
        return f"NormalizeFilter()"
