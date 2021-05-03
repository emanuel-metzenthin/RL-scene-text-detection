import ray


@ray.remote
class ParameterServer:
    def __init__(self):
        self.current_parameters = None

    def publish_parameters(self, object_ref):
        del self.current_parameters
        self.current_parameters = object_ref

    def get_current_parameters(self):
        print(f"current params: {self.current_parameters}")
        return self.current_parameters
