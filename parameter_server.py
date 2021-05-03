import ray


@ray.remote
class ParameterServer:
    def __init__(self):
        self.current_parameters = None

    def publish_parameters(self, object_ref):
        print(f"published params 1")
        if self.current_parameters:
            del self.current_parameters
        print(f"published params 2")
        self.current_parameters = object_ref
        print(f"published params 3 {self.current_parameters})")

    def get_current_parameters(self):
        print(f"current params: {self.current_parameters}")
        return self.current_parameters
