import ray


@ray.remote
class ParameterServer:
    def __init__(self):
        self.current_parameters = None

    def publish_parameters(self, object_ref):
        if self.current_parameters:
            del self.current_parameters
        self.current_parameters = object_ref

    def get_current_parameters(self):
        return self.current_parameters
