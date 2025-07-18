import langchain
from langchain_mcp_adapters.client import MultiServerMCPClient 
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.cache import UpstashRedisCache
from langchain.globals import set_llm_cache


from dotenv import load_dotenv
load_dotenv()

import asyncio
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
from upstash_redis import Redis

import os
import re
import streamlit as st
import time


URL = os.getenv("UPSTASH_REDIS_REST_URL")
TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN")
redis_client = Redis(url=URL, token=TOKEN)
cache = UpstashRedisCache(redis_=redis_client, ttl=3600)  # ttl optional (seconds)
set_llm_cache(cache)

def create_message_template():
    """Create a prompt template for concise Kubernetes status output"""
    template = (
        "Transform this user request into a direct action command. Always use 'staging' namespace.\n"
        "ONLY respond if the request is for:\n"
        "- List resources (pods, deployments, services, etc.)\n"
        "- View/show/get logs from pods/containers\n"
        "- Describe resources\n"
        "- Rollout deployments (restart, status, history)\n"
        "- List/find/extract/view ingress paths and related services\n"
        "- Find services associated with ingress paths\n"
        "- Query ingress configurations and their backend services\n"
        "- Check service/pod status (is X running?, what's the status of X?, is X down?)\n\n"
        "Examples of valid requests:\n"
        "- 'list pods' -> 'list pods in staging'\n"
        "- 'restart browser service' -> 'rollout  browser service deployment in staging'\n"
        "- 'list services for ingress path /api/v1/' -> 'list ingress paths and services for /api/v1/'\n"
        "- 'show logs for pod-name' -> 'get logs from pod-name in staging'\n"
        "- 'is browser service running?' -> 'list pods, check status (running or not) of browser service'\n"
        "- 'is there any service down?' -> 'list all pods in staging with status & summarize'\n"
        "- 'what's the status of api service?' -> 'check status of api service by listing pods  in staging'\n"
        "- 'is payment pod running?' -> 'check status of payment pods by listing pods in staging'\n\n"
        "- 'is payment pod down?' -> 'check status of payment pods by listing pods in staging'\n\n"
        "For anything else (like math, general questions, creating/deleting resources), respond EXACTLY: 'I am not allowed to perform that action.'\n\n"
        "Transform valid requests to direct commands:\n"
        "- Use pod names, not label selectors\n"
        "- For restarts, list the deployment in staging namespace and use rollout \n"
        "- For ingress queries, include path information if provided\n"
        "- For status checks, convert to list and describe commands to check pod/deployment status\n"
        "- Be concise, no explanations\n\n"
        "User request: {user_input}\n\nDirect command:"
    )
    return PromptTemplate(
        template=template,
        input_variables=["user_input"]
    )

def remove_followup_questions(text):
    # Remove common follow-up question patterns at the end of text
    text = re.sub(r"(To assist further,|Could you please|This will allow me|please provide:|What would you like me to do next\?|Is there anything else|Let me know if you need).*$", "", text, flags=re.IGNORECASE)
    # Remove conversational patterns at the end of text
    text = re.sub(r"(Got it!|How can I assist you|Would you like to|Please specify what you need|Let me know how).*$", "", text, flags=re.IGNORECASE)
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
    st.set_page_config(
        page_title="Dokan Cloud K8s Dashboard",
        page_icon="⚙️",
        layout="wide"
    )
    st.title("Kubernetes Assistant")
    st.write("Interact with dokan-cloud staging Kubernetes cluster.")

    # Initialize session state for input field
    if 'user_input' not in st.session_state:
        st.session_state.user_input = "list all pods"

    user_input = st.text_input("Enter your query:", value=st.session_state.user_input, key="input_field")
    
    # Add quick action buttons with black background and white text
    st.markdown("""
    <style>
    .stButton > button {
        background-color: #000000 !important;
        color: white !important;
        border: 1px solid #333333 !important;
        border-radius: 8px !important;
        margin: 0px !important;
        padding: 8px 12px !important;
        font-size: 12px !important;
        height: 40px !important;
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
    .stButton > button:hover {
        background-color: #333333 !important;
        color: white !important;
    }
    div[data-testid="column"] {
        padding: 0px 4px !important;
        margin-bottom: 8px !important;
    }
    /* Style for Submit button - make it different */
    .stButton > button[kind="primary"] {
        background-color: #007ACC !important;
        color: white !important;
        border: 1px solid #005a9e !important;
        border-radius: 8px !important;
        font-size: 14px !important;
        font-weight: bold !important;
        height: 45px !important;
        margin-top: 10px !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #005a9e !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.write("Quick actions:")
    
    # Create multiple rows of buttons for better spacing
    # First row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("list all services"):
            st.session_state.user_input = "list all pods"
            st.rerun()
    
    with col2:
        if st.button("restart a service"):
            st.session_state.user_input = "restart activity service app"
            st.rerun()
    
    with col3:
        if st.button("check service status"):
            st.session_state.user_input = "is user service app running?"
            st.rerun()
    
    # Second row
    col4, col5, col6 = st.columns(3)
    
    with col4:
        if st.button("find service for ingress path"):
            st.session_state.user_input = "find the service associated with this ingress path /api/v1/manifest.json"
            st.rerun()
    
    with col5:
        if st.button("find stopped services"):
            st.session_state.user_input = "is there any service down?"
            st.rerun()
            
    with col6:
        if st.button("get a service logs"):
            st.session_state.user_input = "get logs from activity service"
            st.rerun()

    submit = st.button("Submit", type="primary")

    if submit and user_input.strip():
        with st.spinner("Processing..."):
            # Get the Kubernetes MCP path from environment variable with fallback
            kubernetes_mcp_path = os.getenv("KUBERNETES_MCP_PATH", "/home/appuser/app/kubernetes-mcp-amd64")
            
            # Create a MultiServerMCPClient instance
            client = MultiServerMCPClient(
                {
                    "kubernetes": {
                        "command": kubernetes_mcp_path,
                        "args": [kubernetes_mcp_path],
                        "transport": "stdio",
                        "env" :{
                            "GOOGLE_APPLICATION_CREDENTIALS": "/Users/abdulhannan/Downloads/dokan-dev-cef617497374.json",
                            "GOOGLE_CLOUD_PROJECT": "dokan-dev",
                            "GCP_SECRET_NAME": "dokan-dev-staging-secrets"
                        }
                    },
                }
            )

            os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
            os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

            async def process():
                # Create system prompt for the agent
                system_prompt = (
                    "You are a Kubernetes operations assistant. Execute the requested action using available tools. "
                    "You can list resources, view logs, describe resources, manage rollouts, and list ingress paths with associated services. "
                    "For status checks: use list tools to check pod/deployment status, then respond with simple status. "
                    "If the request is not supported by your tools (like math, general questions, creating/deleting resources), respond EXACTLY: "
                    "'I am not allowed to perform that action.' "
                    "Do not ask questions. Do not offer suggestions. Just execute or reject."
                )

                tools = await client.get_tools()
                from langchain.globals import set_llm_cache
                import langchain
                original_cache = langchain.llm_cache if hasattr(langchain, 'llm_cache') else None
                
                # Use OpenAI for transformation (caching enabled)
                print(f"[CACHE DEBUG] Cache is enabled: {langchain.llm_cache is not None}")
                transform_model = OpenAI(
                    model="gpt-4o-mini",
                )
                start_time = time.time()
                detailed_message = await transform_user_message(user_input, transform_model)
                elapsed = time.time() - start_time
                print(f"[CACHE DEBUG] Transformation step took {elapsed:.3f} seconds.")
                print(f"Original input: {user_input}")
                print(f"Transformed message: {detailed_message}")
                if not detailed_message.strip():
                    detailed_message = user_input.strip()
                
                # Disable cache only for ChatOpenAI agent step
                set_llm_cache(None)
                try:
                    from langchain_openai import ChatOpenAI
                    agent_model = ChatOpenAI(
                        model="gpt-4o-mini",
                    )
                    agent = create_react_agent(agent_model, tools)
                    messages = [
                        ("system", system_prompt),
                        ("user", detailed_message)
                    ]
                    response = await agent.ainvoke({"messages": messages})
                finally:
                    set_llm_cache(original_cache)
                output = response['messages'][-1].content
                output = remove_think_blocks(output)
                output = remove_followup_questions(output)
                return output

            import asyncio
            output = asyncio.run(process())
            st.markdown("**Response:**")
            st.markdown(output, unsafe_allow_html=True)

if __name__ == "__main__":
    main()