import setuptools

setuptools.setup(
    name="imageai",# Replace with your own username
    version="0.0.4",
    author="Andy Jackson",
    author_email="amjack100@gmail.com",
    description="Test",
    long_description_content_type="text/markdown",
    url="None",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    entry_points={
        "console_scripts": [
            "imageai = imageai.cli:cli",
        ],
    },
)