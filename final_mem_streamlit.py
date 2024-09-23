from dotenv import load_dotenv
import os
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnablePassthrough
from groq import Groq
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import StructuredTool
from enum import Enum
from typing import Optional
from langgraph.prebuilt import ToolExecutor
from langchain_groq import ChatGroq
from langchain_core.utils.function_calling import convert_to_openai_function
from typing import TypedDict, Sequence
from langchain_core.messages import BaseMessage
import json
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolInvocation
from langgraph.graph import StateGraph, END
import warnings
import streamlit as st

# Ignore specific deprecation warning
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*ToolInvocation is deprecated.*")

# Load .env file
load_dotenv()

# Set model variables
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "Travel Chatbot"


system_prompt_initial = """
Your job is to assess a brief chat history in order to determine if the conversation contains any details about a family's travel, accomodation, and food preferences. 

You are part of a team building a knowledge base regarding a family's travel, accomodation, and food preferences.

You play the critical role of assessing the message to determine if it contains any information worth recording in the knowledge base.

You are only interested in the following categories of information:

1. The family's food allergies (e.g. a dairy or soy allergy)
2. Food, places, and activities the family likes (e.g. likes pasta, likes going to pubs, likes beaches etc.)
3. Food, places, and activities the family dislikes (e.g. doesn't eat mussels, doesn't like clubs etc.)
4. Attributes about the family that may impact travel itinerary (e.g. has a husband and 2 children, likes to explore local culture, traveling on a budget etc.)

When you receive a message, you perform a sequence of steps consisting of:

1. Analyze the message for information.
2. If it has any information worth recording, return TRUE. If not, return FALSE.

You should ONLY RESPOND WITH TRUE OR FALSE. Absolutely no other information should be provided.

Take a deep breath, think step by step, and then analyze the following message:
"""

# Get the prompt to use - you can modify this!
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(system_prompt_initial),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Remember, only respond with TRUE or FALSE. Do not provide any other information.",
        ),
    ]
)

# Choose the LLM that will drive the agent
llm = ChatGroq(
    model="llama3-groq-70b-8192-tool-use-preview",
    temperature=0.0,
)

sentinel_runnable = {"messages": RunnablePassthrough()} | prompt | llm


class Category(str, Enum):
    Food_Allergy = "Allergy"
    Food_Like = "Like"
    Food_Dislike = "Dislike"
    Family_Attribute = "Attribute"


class Action(str, Enum):
    Create = "Create"
    Update = "Update"
    Delete = "Delete"


class AddKnowledge(BaseModel):
    knowledge: str = Field(
        ...,
        description="Condensed bit of knowledge to be saved for future reference in the format: [person(s) this is relevant to] [fact to store] (e.g. Husband doesn't like tuna; I am allergic to shellfish; etc)",
    )
    knowledge_old: Optional[str] = Field(
        None,
        description="If updating or deleting record, the complete, exact phrase that needs to be modified",
    )
    category: Category = Field(
        ..., description="Category that this knowledge belongs to"
    )
    action: Action = Field(
        ...,
        description="Whether this knowledge is adding a new record, updating a record, or deleting a record",
    )


def modify_knowledge(
    knowledge: str,
    category: str,
    action: str,
    knowledge_old: str = "",
) -> dict:
    print("Modifying Knowledge: ", knowledge, knowledge_old, category, action)
    return "Modified Knowledge"


tool_modify_knowledge = StructuredTool.from_function(
    func=modify_knowledge,
    name="Knowledge_Modifier",
    description="Add, update, or delete a bit of knowledge",
    args_schema=AddKnowledge,
)

# Set up the tools to execute them from the graph

# Set up the agent's tools
agent_tools = [tool_modify_knowledge]

tool_executor = ToolExecutor(agent_tools)

system_prompt_initial = """
You are a supervisor managing a team of knowledge eperts.

Your team's job is to create a perfect knowledge base about a family's or an individual's travel, accomodation, and food preferences.

The knowledge base should ultimately consist of many discrete pieces of information that add up to a rich persona (e.g. I like pasta; I am allergic to shellfish; I like going to pubs; I like beaches; I like to explore local culture; traveling on a budget etc.)

Every time you receive a message, you will evaluate if it has any information worth recording in the knowledge base.

A message may contain multiple pieces of information that should be saved separately.

You are only interested in the following categories of information:

1. The family's food allergies (e.g. a dairy or soy allergy) - These are important to know because they can be life-threatening. Only log something as an allergy if you are certain it is an allergy and not just a dislike.
2. Food, places, and activities the family likes (e.g. likes pasta, likes going to pubs, likes beaches etc.) - These are important to know because they can help you plan meals, and places to visit but are not life-threatening.
3. Food, places, and activities the family dislikes (e.g. doesn't eat mussels, doesn't like clubs etc.) - These are important to know because they can help you plan meals, and places to visit but are not life-threatening.
4. Attributes about the family that may impact travel itinerary (e.g. has a husband and 2 children, likes to explore local culture, traveling on a budget etc.)

When you receive a message, you perform a sequence of steps consisting of:

1. Analyze the most recent Human message for information. You will see multiple messages for context, but we are only looking for new information in the most recent message.
2. Compare this to the knowledge you already have.
3. Determine if this is new knowledge, an update to old knowledge that now needs to change, or should result in deleting information that is not correct. It's possible that a food you previously wrote as a dislike might now be a like, or that a family member who previously liked a food now dislikes it - those examples would require an update.

Here are the existing bits of information that we have about the family.

```
{memories}
```

Call the right tools to save the information, then respond with DONE. If you identiy multiple pieces of information, call everything at once. You only have one chance to call tools.

I will tip you $20 if you are perfect, and I will fine you $40 if you miss any important information or change any incorrect information.

Take a deep breath, think step by step, and then analyze the following message:
"""

# Get the prompt to use - you can modify this!
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(system_prompt_initial),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Choose the LLM that will drive the agent
llm = ChatGroq(
    model="llama3-groq-70b-8192-tool-use-preview",
    temperature=0.0,
)


# Create the tools to bind to the model
tools = [convert_to_openai_function(t) for t in agent_tools]

knowledge_master_runnable = prompt | llm.bind_tools(tools)


class AgentState(TypedDict):
    # The list of previous messages in the conversation
    messages: Sequence[BaseMessage]
    # The long-term memories to remember
    memories: Sequence[str]
    # Whether the information is relevant
    contains_information: str


def call_sentinel(state):
    messages = state["messages"]
    response = sentinel_runnable.invoke(messages)
    return {"contains_information": "TRUE" in response.content and "yes" or "no"}


# Define the function that determines whether to continue or not
def should_continue(state):
    last_message = state["messages"][-1]
    # If there are no tool calls, then we finish
    if "tool_calls" not in last_message.additional_kwargs:
        return "end"
    # Otherwise, we continue
    else:
        return "continue"


# Define the function that calls the knowledge master
def call_knowledge_master(state):
    messages = state["messages"]
    memories = state.get("memories", [])
    response = knowledge_master_runnable.invoke(
        {"messages": messages, "memories": memories}
    )
    return {"messages": messages + [response]}


# Define the function to execute tools
def call_tool(state):
    messages = state["messages"]
    # We know the last message involves at least one tool call
    last_message = messages[-1]

    # We loop through all tool calls and append the message to our message log
    for tool_call in last_message.additional_kwargs["tool_calls"]:
        action = ToolInvocation(
            tool=tool_call["function"]["name"],
            tool_input=json.loads(tool_call["function"]["arguments"]),
            id=tool_call["id"],
        )

        # We call the tool_executor and get back a response
        response = tool_executor.invoke(action)
        # We use the response to create a FunctionMessage
        function_message = ToolMessage(
            content=str(response), name=action.tool, tool_call_id=tool_call["id"]
        )

        # Add the function message to the list
        messages.append(function_message)
    return {"messages": messages}



# Initialize a new graph
graph = StateGraph(AgentState)

# Define the two "Nodes"" we will cycle between
graph.add_node("sentinel", call_sentinel)
graph.add_node("knowledge_master", call_knowledge_master)
graph.add_node("action", call_tool)

# Define all our Edges

# Set the Starting Edge
graph.set_entry_point("sentinel")

# We now add Conditional Edges
graph.add_conditional_edges(
    "sentinel",
    lambda x: x["contains_information"],
    {
        "yes": "knowledge_master",
        "no": END,
    },
)
graph.add_conditional_edges(
    "knowledge_master",
    should_continue,
    {
        "continue": "action",
        "end": END,
    },
)

# We now add Normal Edges that should always be called after another
graph.add_edge("action", END)

# We compile the entire workflow as a runnable
app = graph.compile()

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# Modify the chatbot function to work with Streamlit
def chatbot(user_input, model):
    previous_memories = st.session_state.get('previous_memories', [])
    
    inputs = {
        "messages": [HumanMessage(content=user_input)],
        "memories": previous_memories
    }
    formatted_prompt = """
    You are a travel AI assistant, you will only respond after considering the user prompt and the previous memories you have of the user.
    Here are the previous memories:
    {previous_memories}
    Here is the user prompt:
    {user_input}
    """
    messages = [{"role": "user", "content": formatted_prompt.format(previous_memories=previous_memories, user_input=user_input)}]
    response = client.chat.completions.create(
        model=model,
        messages=messages, 
    )
    answer = response.choices[0].message.content

    actions = []
    for output in app.with_config({"run_name": "Memory"}).stream(inputs):
        for key, value in output.items():
            if key == 'knowledge_master':
                actions.extend(value['messages'][1].additional_kwargs['tool_calls'])

    new_memories = []
    for action in actions:
        arguments = json.loads(action['function']['arguments'])
        if arguments['action'] == 'Create':
            memory_entry = {
                "category": arguments['category'],
                "knowledge": arguments['knowledge']
            }
            new_memories.append(memory_entry)
    
    st.session_state.previous_memories = previous_memories + new_memories
    return answer, new_memories

model = "mixtral-8x7b-32768"
# Streamlit UI
st.title("Crypto Xpress Travel AI Assistant")
# Display the current model in small, gray text
st.markdown(f'<p style="color:gray;font-size:12px;">Current model: {model}</p>', unsafe_allow_html=True)

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'previous_memories' not in st.session_state:
    st.session_state.previous_memories = []

# Display chat history
st.subheader("Chat History")
chat_container = st.container()

with chat_container:
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(message)

# User input
user_input = st.chat_input("Type your message here...", key="user_input")

if user_input:
    # Add user message to chat history
    st.session_state.chat_history.append(("You", user_input))
    
    # Get chatbot response
    response, new_memories = chatbot(user_input, model)
    
    # Add chatbot response to chat history
    st.session_state.chat_history.append(("Assistant", response))
    
    # Update UI
    with chat_container:
        with st.chat_message("You"):
            st.write(user_input)
        with st.chat_message("Assistant"):
            st.write(response)
    
    # Display new memories
    if new_memories:
        st.subheader("New Memories:")
        for memory in new_memories:
            st.write(f"**{memory['category']}:** {memory['knowledge']}")

# Display all memories
with st.expander("All Memories", expanded=False):
    for memory in st.session_state.previous_memories:
        st.write(f"**{memory['category']}:** {memory['knowledge']}")
