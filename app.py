import asyncio
import os
import sys
from dotenv import load_dotenv
# from langchain_community.llms import Ollama
from langchain_groq import ChatGroq
from mcp_use import MCPAgent, MCPClient


async def main():
    # Load environment variables
    load_dotenv()
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

    print("Initializing your AI Assistant...", end='\n\n')

    # Create MCPClient from config file
    client = MCPClient.from_config_file(os.path.join(os.path.dirname(__file__), "mcp.json"))

    # Create LLM using Ollama
    llm = ChatGroq(model="qwen-qwq-32b")

    # Create the MCPAgent
    agent = MCPAgent(llm=llm, client=client, max_steps=30)

    print("Your AI Assistant is ready to help you. Type 'exit' to quit.\n\n")

    try:
        while True:
            user_input = input("Enter a query: ")
            if user_input.lower() == "exit":
                print("Exiting...")
                break
            result = await agent.run(user_input)
            print(f"\nAssistant: {result}")
    finally:
        if client.sessions:
            await client.close_all_sessions()
            print("All sessions closed")

if __name__ == "__main__":
    asyncio.run(main())
    #go run main.go
