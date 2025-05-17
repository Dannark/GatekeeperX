from setuptools import setup, find_packages

setup(
    name="gatekeeperx",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "opencv-python",
        "ultralytics",
        "numpy"
    ]
) 