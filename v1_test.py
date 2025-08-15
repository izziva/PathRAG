import os
from PathRAG import PathRAG, QueryParam
from PathRAG.llm import litellm_completion
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

WORKING_DIR = ""


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# Since litellm_completion is now the default, we don't need to pass it explicitly.
# However, to keep the structure of this test file, we will pass it.
rag = PathRAG(
    working_dir=WORKING_DIR,
    llm_model_func=litellm_completion,
)

data_file=""
question=""
with open(data_file) as f:
    rag.insert(f.read())

print(rag.query(question, param=QueryParam(mode="hybrid")))














