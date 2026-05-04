import os
from typing import Optional

import cv2
import ollama

from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage


load_dotenv()

model_name = "moondream:1.8b"

@tool
def analyze_robot_vision(query: str, image_path: Optional[str] = None) -> str:
    """
    Use this tool to see the world. 
    If 'image_path' is provided, the robot looks at that specific file.
    Otherwise, it uses its live camera feed.
    """
    print(f"\n[System] Brain requested Vision. Query: '{query}'")
    
    frame = None
    
    # Check if we should use a specific file or the live camera
    if image_path and os.path.exists(image_path):
        print(f"[System] Loading image from path: {image_path}")
        frame = cv2.imread(image_path)
    else:
        print("[System] Using live camera feed.")
        cap = cv2.VideoCapture(0)
        for _ in range(10):
            cap.read()
        ret, captured_frame = cap.read()
        if ret:
            frame = captured_frame
        cap.release()

    if frame is not None:
        _, buffer = cv2.imencode(".jpg", frame)
        image_bytes = buffer.tobytes()

        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": query if query else "What do you see?",
                    "images": [image_bytes],
                }
            ],
        )
        return response["message"]["content"]
    
    return "I tried to see, but I couldn't access the image source."

# Initialize the LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)

# llm = ChatOllama(
#     model="qwen3.5:4b",
#     temperature=0,
# )

tools = [analyze_robot_vision]


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are the AI brain of Reachy, a humanoid robot who would work in an AI/Robotics lab. Use your tools to see the world."
            "Report only high-confidence visual facts. Do not speculate on the purpose or condition of objects unless explicitly visible",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)


agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


print("\n--- Reachy is Online ---")
print("Type 'exit' or 'quit' to stop the session.\n")

chat_history = []

while True:
    user_input = input("You: ")
    
    if user_input.lower() in ["exit", "quit"]:
        print("Shutting down Reachy's brain...")
        break

    try:
        response = agent_executor.invoke({
            "input": user_input,
            "chat_history": chat_history
        })
        
        output = response["output"]
        print(f"\nReachy: {output}\n")

        # 4. Update Memory
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=output))

    except Exception as e:
        print(f"\n[Error] Brain hiccup: {e}\n")

