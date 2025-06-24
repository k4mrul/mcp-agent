from langchain_mcp_adapters.client import MultiServerMCPClient 
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
load_dotenv()

import asyncio
from langchain_groq import ChatGroq
import re

def create_message_template():
    """Create a prompt template for concise Kubernetes status output"""
    template = """You are a Kubernetes assistant. Given the user request, first try to list all the pods in the staging namespace. from there, try to find the name of that pod. if pod exist, list it. if not, say pod not found.

User request: {user_input}

Output:"""
    return PromptTemplate(
        template=template,
        input_variables=["user_input"]
    )

async def transform_user_message(user_input: str, model) -> str:
    """Transform user input using LangChain prompt template with fallback for common patterns"""
    user_input_lower = user_input.lower().strip()
    
    # Quick pattern matching for common cases to avoid unnecessary LLM calls
    if "staging namespace" in user_input_lower:
        # Already has staging namespace context, return as is
        return user_input
    
    # For service status queries, use the prompt template to get detailed instructions
    if "service" in user_input_lower and "status" in user_input_lower:
        prompt_template = create_message_template()
        chain = prompt_template | model | StrOutputParser()
        
        try:
            result = await chain.ainvoke({"user_input": user_input})
            return result.strip()
        except Exception as e:
            print(f"Template transformation failed: {e}")
            return f"{user_input} in staging namespace"
    
    # For simple service queries without status, add namespace context manually
    # if "service" in user_input_lower:
    #     return f"{user_input} in staging namespace"
    
    # For pod queries, add namespace context
    if any(keyword in user_input_lower for keyword in ["pods", "pod status"]):
        return f"{user_input} in staging namespace"
    
    # For other cases, use the prompt template
    prompt_template = create_message_template()
    chain = prompt_template | model | StrOutputParser()
    
    try:
        result = await chain.ainvoke({"user_input": user_input})
        return result.strip()
    except Exception as e:
        # Fallback to original message if template fails
        print(f"Template transformation failed: {e}")
        return f"{user_input} in staging namespace"

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


    tools = await client.get_tools()
    model=ChatGroq(
        model="qwen-qwq-32b",
    )   
    
    # Original user input
    user_input = "what is the status of bilingual service. is it running?"
    
    # Process the user input through the template to get detailed instructions
    detailed_message = await transform_user_message(user_input, model)
    detailed_message = remove_think_blocks(detailed_message)
    print(f"Original message: {user_input}")
    # print(f"Enhanced message: {detailed_message}")
    # print("-" * 50)
    
    # Create a React agent using the client
    agent = create_react_agent(model, tools)

    # Run the agent with the enhanced message
    response = await agent.ainvoke({"messages": [("user", detailed_message)]})

    output = response['messages'][-1].content
    output = remove_think_blocks(output)
    print(output)

# Run the main function
asyncio.run(main())