from langchain_mcp_adapters.client import MultiServerMCPClient 
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
load_dotenv()

import asyncio
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI

import re

def create_message_template():
    """Create a prompt template for concise Kubernetes status output"""
    template = (
        "You are a Kubernetes assistant. Look for pods in the 'staging' namespace."
        "Also, look for pod name, don't use label selector. "
        "Use short answers, no follow-up questions, and no explanations.\n\n"
        "User request: {user_input}\n\nOutput:"
    )
    return PromptTemplate(
        template=template,
        input_variables=["user_input"]
    )

def remove_followup_questions(text):
    # Remove common follow-up question patterns
    text = re.sub(r"(To assist further,|Could you please|This will allow me|please provide:|\b1\.|\b2\.).*", "", text, flags=re.IGNORECASE|re.DOTALL)
    # Remove trailing whitespace and extra punctuation
    return text.strip().rstrip('.')

async def transform_user_message(user_input: str, model) -> str:
    """Transform user input using LangChain prompt template with fallback for common patterns"""
    prompt_template = create_message_template()
    chain = prompt_template | model | StrOutputParser()
    try:
        result = await chain.ainvoke({"user_input": user_input})
        result = result.strip().replace("\n", " ")
        result = remove_followup_questions(result)
        return result
    except Exception as e:
        print(f"Template transformation failed: {e}")
        # Fallback: return the original user input, stripped and single line
        return remove_followup_questions(user_input.strip().replace("\n", " "))

def remove_think_blocks(text):
    # Remove <think>...</think> blocks (including multiline)
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

async def main():
    # Create a MultiServerMCPClient instance
    client = MultiServerMCPClient(
        {
            "kubernetes": {
                "command": "/Users/abdulhannan/Desktop/ai/mcp/kubernetes-mcp/kubernetes-mcp",
                "args": ["/Users/abdulhannan/Desktop/ai/mcp/kubernetes-mcp/kubernetes-mcp"],
                "transport": "stdio",
                "env" :{
                    "HOME": "/Users/abdulhannan",
                    "GOOGLE_APPLICATION_CREDENTIALS": "/Users/abdulhannan/Downloads/dokan-dev-cef617497374.json",
                    "GOOGLE_CLOUD_PROJECT": "dokan-dev",
                    "GCP_SECRET_NAME": "dokan-dev-staging-secrets"
                }
            }
        }
    )

    import os
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


    tools = await client.get_tools()
    # model=ChatGroq(
    #     model="llama3-70b-8192",
    # )   
    model=ChatOpenAI(
        model="gpt-4o-mini",
    )
    # model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    
    # Original user input
    # user_input = "what's the status of activity pods"
    # user_input = "is the report service running?"
    # user_input = "restart activity service pod"
    user_input = "what is the pods logs for activity service"
    
    # Process the user input through the template to get detailed instructions
    detailed_message = await transform_user_message(user_input, model)
    # Fallback if enhanced message is empty
    if not detailed_message.strip():
        print("Warning: Enhanced message was empty, falling back to user input.")
        detailed_message = user_input.strip()
    # detailed_message = remove_think_blocks(detailed_message)
    print(f"Original message: {user_input}")
    print(f"Enhanced message: {detailed_message}")
    print("-" * 50)
    
    # Create a React agent using the client
    agent = create_react_agent(model, tools)

    # Run the agent with the enhanced message
    response = await agent.ainvoke({"messages": [("user", detailed_message)]})

    output = response['messages'][-1].content
    output = remove_think_blocks(output)
    print(output)

# Run the main function
asyncio.run(main())