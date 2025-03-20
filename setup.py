from setuptools import setup, find_packages

setup(
    name="openai_cost_calculator",
    version="0.1.1",
    description="A library to estimate OpenAI API costs based on token usage.",
    author="Orkun Kınay, Murat Barkın Kınay",
    author_email="orkunkinay@sabanciuniv.edu",
    url="https://github.com/orkunkinay/openai_cost_calculator", 
    packages=find_packages(),
    include_package_data=True, 
    package_data={
        "openai_cost_calculator": ["data/gpt_pricing_data.csv"],
    },
    install_requires=["requests"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
