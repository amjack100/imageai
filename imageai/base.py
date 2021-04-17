import abc


class ImgModule(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "hub_module")
            and callable(subclass.run)
            and callable(subclass._preprocess)
            or NotImplemented
        )

    # hub_module = ""
    @abc.abstractmethod
    def _preprocess(self):

        raise NotImplementedError

    @abc.abstractmethod
    def run(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _save_results(self):
        raise NotImplementedError