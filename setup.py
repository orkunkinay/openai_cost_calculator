from setuptools import setup, find_packages

setup(
    name="openai_cost_calculator",
    version="1.2.0",
    description="A library to estimate OpenAI API costs based on token usage.",
    author="Orkun Kınay, Murat Barkın Kınay",
    author_email="orkunkinay@sabanciuniv.edu",
    url="https://github.com/orkunkinay/openai_cost_calculator", 
    packages=find_packages(),
    include_package_data=True, 
    package_data={
        "openai_cost_calculator": ["data/gpt_pricing_data.csv"],
    },
    install_requires=["requests", "tomli; python_version < '3.11'"],
    extras_require={
        "proxy": ["fastapi", "httpx", "uvicorn", "websockets>=13"],
    },
    entry_points={
        "console_scripts": [
            "openai-cost-calculator=openai_cost_calculator.cli:main",
            "occ-cc-statusline=openai_cost_calculator.adapters.claude_code:statusline_main",
            "occ-cc-stop-hook=openai_cost_calculator.adapters.claude_code:stop_hook_main",
            "occ-codex-notify=openai_cost_calculator.adapters.codex:notify_main",
            "occ-codex-statusline=openai_cost_calculator.adapters.codex:statusline_main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)
