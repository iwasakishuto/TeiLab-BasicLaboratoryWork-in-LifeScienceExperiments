# coding: utf-8
import os
import setuptools
import teilab

DESCRIPTION = "シークエンシング、生化学反応シミュレーション、および遺伝子発現プロファイリング等の生物情報科学的実験を行う。本実習は、生物化学科と合同で実施する。"

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()
with open("requirements.txt", mode="r") as f:
    INSTALL_REQUIRES = [line.rstrip("\n") for line in f.readlines()]

def setup_package():
    metadata = dict(
        name="UiTei-Lab-Course",
        version=teilab.__version__,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        author=teilab.__author__,
        author_email=teilab.__author_address__,
        license=teilab.__license__,
        project_urls={
            "Documentation" : teilab.__documentation__,
            "Bug Reports"   : "https://github.com/iwasakishuto/TeiLab-BasicLaboratoryWork-in-LifeScienceExperiments/issues",
            "Source Code"   : "https://github.com/iwasakishuto/TeiLab-BasicLaboratoryWork-in-LifeScienceExperiments",
            "Say Thanks!"   : "https://twitter.com/cabernet_rock",
        },
        packages=setuptools.find_packages(),
        python_requires="3.8.9",
        install_requires=INSTALL_REQUIRES,
        extras_require={
          "tests": ["pytest"],
        },
        classifiers=[
            "Development Status :: 3 - Alpha",
            "License :: OSI Approved :: MIT License",
            "License :: Free For Educational Use",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Topic :: Education",
            "Topic :: Software Development :: Libraries",
            "Topic :: Software Development :: Libraries :: Python Modules",
            "Intended Audience :: Education",
        ],
        entry_points = {},
    )
    setuptools.setup(**metadata)

if __name__ == "__main__":
    setup_package()