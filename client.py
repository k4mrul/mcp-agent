from langchain_mcp_adapters.client import MultiServerMCPClient 
from langgraph.prebuilt import create_react_agent

from dotenv import load_dotenv
load_dotenv()

import asyncio
from langchain_groq import ChatGroq

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
    # Create a React agent using the client
    agent = create_react_agent(model, tools)

    # Run the agent with a sample input
    # response = await agent.run("get pods in staging namespace")
    response = await agent.ainvoke({"messages": [("user", "get pods in staging namespace")]})

    print(response['messages'][-1].content)

# Run the main function
asyncio.run(main())