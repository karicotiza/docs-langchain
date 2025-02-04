# docs-langchain

Here I write code based on the LangChain documentation.

## How to reproduce

Prerequisites:

1. Python `3.12.3`
2. Docker Engine `27.4.0`
3. Docker Compose `2.31.0`

Steps:

1. Install dependencies -
`python -m pip install -r requirements.txt`
2. Spin up the required infrastructure using Docker -
`docker compose -f ./build/prod.yml up --build --detach`
3. Run tests -
`python -m pytest`
