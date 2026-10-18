"""Optional preparation of a new token stream using the pinned dataset reader."""
import runpy
from .dataset_compat import enable_list_reader


if __name__ == "__main__":
    enable_list_reader()
    runpy.run_module("experiment.prepare", run_name="__main__")
