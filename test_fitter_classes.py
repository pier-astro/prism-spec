from astropy.modeling.fitting import Fitter
import inspect

def patch_fitters():
    def get_all_subclasses(cls):
        all_subclasses = []
        for subclass in cls.__subclasses__():
            all_subclasses.append(subclass)
            all_subclasses.extend(get_all_subclasses(subclass))
        return all_subclasses

    target_classes = [Fitter] + get_all_subclasses(Fitter)
    for cls in target_classes:
        if '__call__' in cls.__dict__:
            print(f"Patching {cls.__name__}.__call__")

patch_fitters()
