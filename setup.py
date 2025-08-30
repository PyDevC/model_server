import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="model_server",  # Replace with your desired package name
    version="0.1.0",
    author="PyDevC",
    author_email="anuskcse@gmail.com",
    description="LLM inference tool for offline huggingface models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="[https://github.com/PyDevC/model_server](https://github.com/PyDevC/model_server)",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "model_server=model_server.maininference:start",
        ],
    },
)
