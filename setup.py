from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
        name='plotstyles',
        version='0.0.1',
        description='This is a package for personal plot',
        author='xfw',
        author_email='xfwahss@qq.com',
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/Happy-Grass/plotstyles",
        classifies=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            ],
        python_requires=">=3.6"
        )
