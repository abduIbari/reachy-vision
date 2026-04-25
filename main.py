import ollama

import cv2
import ollama
from io import BytesIO
from dotenv import load_dotenv


load_dotenv()

model_name = "moondream:1.8b"

# cap = cv2.VideoCapture(0)
# ret, frame = cap.read()

# if ret:
#     # Encode frame to JPEG format in memory
#     _, buffer = cv2.imencode(".jpg", frame)
#     image_bytes = buffer.tobytes()

#     response = ollama.chat(
#         model=model_name,
#         messages=[
#             {
#                 "role": "user",
#                 "content": "what do you see?",
#                 "images": [image_bytes],  # Ollama accepts raw bytes
#             }
#         ],
#     )
#     print(response["message"]["content"])

# cap.release()


import os
import ollama
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

# Set your Groq API Key
os.environ["GROQ_API_KEY"] = "your_groq_api_key_here"


# --- STEP 1: Define the Vision Tool ---
@tool
def analyze_robot_vision(query: str) -> str:
    """
    Use this tool ONLY when the user asks questions about the physical environment,
    objects they are holding, or what the robot can see right now.
    """
    print(f"\n[System] Brain requested Vision. Query: '{query}'")

    # Path to the frame captured from Reachy's camera
    # For testing, you can use a static image path

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    if ret:
        # Encode frame to JPEG format in memory
        _, buffer = cv2.imencode(".jpg", frame)
        image_bytes = buffer.tobytes()

        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": "what do you see?",
                    "images": [image_bytes],  # Ollama accepts raw bytes
                }
            ],
        )
        result = response["message"]["content"]
        print(result)
        return result

    cap.release()


# --- STEP 2: Initialize the Groq Brain ---
# llm = ChatGroq(
#     model="openai/gpt-oss-120b",
#     temperature=0,
# )

llm = ChatOllama(
    model="qwen3.5:4b",
    temperature=0,
)

tools = [analyze_robot_vision]

# --- STEP 3: Setup the Agent ---
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are the AI brain of Reachy, a humanoid robot. Use your tools to see the world.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- STEP 4: Run a Test ---
print("Robot Initialized.")
query = "I am holding an object in front of your camera. Can you tell me what it is and what color it is?"
agent_executor.invoke({"input": query})
