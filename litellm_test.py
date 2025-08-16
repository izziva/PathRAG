import asyncio
import os
from PathRAG import PathRAG, QueryParam

# It's important to load environment variables from .env before initializing PathRAG
from dotenv import load_dotenv
load_dotenv()

async def main():
    # Define a working directory for the test
    working_dir = "./test_run_cache"
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)

    print("Initializing PathRAG...")
    # Initialize PathRAG. It will use LiteLLM by default now.
    rag = PathRAG(working_dir=working_dir)

    # Simple data to insert
    data = "LiteLLM is a library that provides a uniform interface for interacting with different large language models."
    print(f"Inserting data: '{data}'")
    await rag.ainsert(data)
    print("Data insertion complete.")

    # A question related to the inserted data
    question = "What is LiteLLM?"
    print(f"Querying with: '{question}'")

    # Run the query
    response = await rag.aquery(question, param=QueryParam(mode="hybrid"))

    print("\n--- Query Response ---")
    print(response)
    print("--- End of Response ---")

    # Verify that the response is not empty
    if response and isinstance(response, str) and response.strip():
        print("Test PASSED: Received a non-empty string answer.")
    else:
        print(f"Test FAILED: Received an invalid or empty response: {response}")

if __name__ == "__main__":
    # In order to run this test, you need to have a .env file in the root directory
    # with your GEMINI_API_KEY.
    # Example .env file:
    # LITELLM_PROVIDER=gemini
    # LITELLM_MODEL=gemini/gemini-1.5-flash-latest
    # LITELLM_EMBEDDING_MODEL=embedding-001
    # GEMINI_API_KEY=YOUR_GEMINI_API_KEY

    asyncio.run(main())
