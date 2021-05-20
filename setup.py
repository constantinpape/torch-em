import runpy
from setuptools import setup, find_packages

__version__ = runpy.run_path("torch_em/__version__.py")["__version__"]


# NOTE requirements are not all available via pip, you need to use conda,
# see 'environment_gpu.yaml' / 'environment_cpu.yaml'
requires = [
    "torch",
    "h5py"
]


setup(
    name="torch_em",
    packages=find_packages(exclude=["test"]),
    version=__version__,
    author="Constantin Pape",
    install_requires=requires,
    url="https://github.com/constantinpape/torch-em",
    license="MIT",
    entry_points={
        "console_scripts": [
            "torch_em.export_bioimageio_model = torch_em.util.modelzoo.main"
        ]
    }
)
