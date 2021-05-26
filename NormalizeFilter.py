from ray.rllib.utils.filter import Filter


class NormalizeFilter(Filter):
    """
    Filter that normalizes
    """

    def apply_changes(self, other, *args, **kwargs):
        pass

    def sync(self, other):
        pass

    def copy(self):
        return NormalizeFilter()

    def as_serializable(self):
        return self.copy()

    def clear_buffer(self):
        pass

    def __call__(self, x, update=True):
        return x / 255

    def __repr__(self):
        return f"NormalizeFilter()"
