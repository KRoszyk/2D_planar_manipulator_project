import abc


class Controller(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def calculate_control(self, *args):
        pass
