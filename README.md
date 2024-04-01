# langchain_experiments

Repo to experiment with different features of langchain library.

**Note:** The experiments expects that GPT model is deployed on microsoft azure.

## Pre-requisite
- python 3.11

## How to setup

### Setup virtual environment

```shell
python -m venv .venv
```

```shell
source .venv/bin/activate
```

### Install dependencies

```shell
pip install -r requirements.txt
```

### Create .env file
Create `.env` file and set the appropriate values.

```dotenv
OPENAI_API_KEY=""
AZURE_OPENAI_URL_BASE=""
AZURE_OPENAI_API_VERSION=""
OPENAI_ORGANIZATION_KEY=""
AZURE_35_DEPLOYMENT_NAME=""
```

### How to run experiments

```shell
# python ./experiments/<experiment_script_python_file_name>
python ./experiments/tools_agent_for_maths.py
```
