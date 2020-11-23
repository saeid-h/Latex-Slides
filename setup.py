import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="latex-slides", # Replace with your own username
    version="0.0.1",
    author="Saeid Hosseinipoor",
    author_email="shossei1@stevens.edu",
    description="A package to make slides for repeating samples.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/saeid-h/latex_slide_maker",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)