from langchain.agents import tool
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from dotenv import load_dotenv
from langchain.tools.render import render_text_description
from langchain.agents.output_parsers.react_single_input import (
    ReActSingleInputOutputParser,
)
from textwrap import dedent
from langchain.schema import AgentAction, AgentFinish
from typing import Union
from langchain.agents.format_scratchpad.log import format_log_to_str
from callbacks import AgentCallbackHandler

load_dotenv()


@tool
def get_text_length(inputString):
    """Returns the length of text by charactersx"""
    return len(inputString)

@tool
def append_text(inputString1):
    """Appends one string to another"""
    return inputString1 + " "


def findToolByName(tools, toolName) -> Tool:
    for tool in tools:
        if tool.name == toolName:
            return tool
    raise ValueError(f"Tool wtih name {toolName} not found")


if __name__ == "__main__":
    print("Hello lang Chain")
    tools = [get_text_length, append_text]
    template = dedent(
        """\
    Answer the following questions as best you can. You have access to the following tools:
 
    {tools}
 
    Use the following format:
 
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
 
    Begin!
 
    Question: {input}
    Thought: {agent_scratchpad}\
    """
    )

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )
    intermediate_steps = []

    llm = ChatOpenAI(temperature=0, stop=["\nObservation"], callbacks = [AgentCallbackHandler()])
    agent = (
        {"input": lambda x: x["input"], "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"])} | prompt | llm | ReActSingleInputOutputParser()
    )

    agentStep = ""

    while not isinstance(agentStep, AgentFinish):

        agentStep: Union[AgentAction, AgentFinish] = agent.invoke(
            {
                "input": "What is the lenght of word : DOG?",
                "agent_scratchpad": intermediate_steps,
            }
        )
        

        if isinstance(agentStep, AgentAction):
            toolName = agentStep.tool
            toolInput = agentStep.tool_input

            toolToUse = findToolByName(tools, toolName)

            observation = toolToUse.func(toolInput)
            intermediate_steps.append((agentStep, str(observation)))

    if isinstance(agentStep, AgentFinish):
        print(agentStep.return_values) 