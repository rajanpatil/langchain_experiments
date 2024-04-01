import os
from typing import Type

# import from .env
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel

load_dotenv()


# Mathematics tools
class MultiplyToolInput(BaseModel):
    first_int: int
    second_int: int


class Multiply(BaseTool):
    name = "multiply"
    description = "Multiply two integers together."
    args_schema: Type[BaseModel] = MultiplyToolInput

    def _run(self, first_int: int, second_int: int) -> int:
        return first_int * second_int

    async def _arun(self, first_int: int, second_int: int) -> int:
        raise NotImplementedError


class AddToolInput(BaseModel):
    first_int: int
    second_int: int


class Add(BaseTool):
    name = "add"
    description = "Add two integers."
    args_schema: Type[BaseModel] = AddToolInput

    def _run(self, first_int: int, second_int: int) -> int:
        return first_int + second_int

    async def _arun(self, first_int: int, second_int: int) -> int:
        raise NotImplementedError


class ExponentiateToolInput(BaseModel):
    base: int
    exponent: int


class Exponentiate(BaseTool):
    name = "exponentiate"
    description = "Exponentiate the base to the exponent power.."
    args_schema: Type[BaseModel] = ExponentiateToolInput

    def _run(self, base: int, exponent: int) -> int:
        return base ** exponent

    async def _arun(self, base: int, exponent: int) -> int:
        raise NotImplementedError


tools = [Add(), Multiply(), Exponentiate()]

# Get the prompt to use - you can modify this!
prompt = ChatPromptTemplate.from_messages(
    [SystemMessagePromptTemplate.from_template("You are a helpful assistant"),
     HumanMessagePromptTemplate.from_template("{input}")
     ])

# Choose the LLM that will drive the agent
model = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_URL_BASE"),
    openai_api_key=os.getenv("OPENAI_ORGANIZATION_KEY"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    deployment_name=os.getenv("AZURE_35_DEPLOYMENT_NAME"),
    model_kwargs={"tools": [convert_to_openai_tool(tool) for tool in tools], "tool_choice": None}
)

# chain = LLMChain(llm=model, prompt=prompt, verbose=True)
# response = chain.invoke(
#     {
#         "input": "Take 3 to the fifth power and multiply that by the sum of twelve and three, then square the whole result"
#     }
# )

response = model.invoke(
    "Take 3 to the fifth power and multiply that by the sum of twelve and three, then square the whole result")

print(response)
