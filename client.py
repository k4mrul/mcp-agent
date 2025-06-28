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
import streamlit as st

def create_message_template():
    """Create a prompt template for concise Kubernetes status output"""
    template = (
        "You are a Kubernetes assistant. Always use the 'staging' namespace."
        "For pods, look for pod name, don't use label selector."
        "When searching for a pod, if the user provides a service name, search for a deployment whose name starts with the requested text (e.g., 'activity-service'). Then, use the pods created by that deployment."
        "Use short answers, no questions, and no explanations. Always proceed the work. Don't wait for user confirmation."
        "If you find a similar deployment or pod, always proceed and explicitly say 'Yes, proceeding with <deployment or pod name>' without asking for user confirmation.\n\n"
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

def main():
    st.title("Kubernetes Assistant")
    st.write("Interact with your Kubernetes cluster using natural language.")

    user_input = st.text_input("Enter your request:", "list all the service in staging namespace")
    submit = st.button("Submit")

    if submit and user_input.strip():
        with st.spinner("Processing..."):
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
                    },
                }
            )

            import os
            os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
            os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

            async def process():
                tools = await client.get_tools()
                model=ChatOpenAI(
                    model="gpt-4o-mini",
                )
                detailed_message = await transform_user_message(user_input, model)
                if not detailed_message.strip():
                    detailed_message = user_input.strip()
                agent = create_react_agent(model, tools)
                response = await agent.ainvoke({"messages": [("user", detailed_message)]})
                output = response['messages'][-1].content
                output = remove_think_blocks(output)
                return output

            import asyncio
            output = asyncio.run(process())
            st.markdown("**Response:**")
            st.markdown(output, unsafe_allow_html=True)

if __name__ == "__main__":
    main()