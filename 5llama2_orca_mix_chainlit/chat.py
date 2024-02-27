from ctransformers import AutoModelForCausalLM
from typing import List
import chainlit as cl

# Function to get the prompt for the model
def get_prompt(instruction: str, history: List[str] = None) -> str:
    system = "You are an AI assistant that gives helpful answers. You answer the questions in a short and concise manner."
    prompt = f"\n\n### System:\n{system}\n\n### User:\n"
    if history and len(history) > 0:
        prompt += f"This is the conversation history: {''.join(history)}. Now answer the question: "
    prompt += f"{instruction}\n\n### Response:\n"
    return prompt

# Function to switch models
def switch_model(model_name: str) -> None:
    global llm
    if model_name.lower() == "llama2":
        llm = AutoModelForCausalLM.from_pretrained(
            "TheBloke/Llama-2-7B-Chat-GGUF",
            model_file="llama-2-7b-chat.Q8_0.gguf"
        )
        print("Model changed to Llama")
    elif model_name.lower() == "orca":
        llm = AutoModelForCausalLM.from_pretrained(
            "zoltanctoth/orca_mini_3B-GGUF",
            model_file="orca-mini-3b.q4_0.gguf"
        )
        print("Model changed to Orca")

# Function to handle incoming messages
@cl.on_message
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history")
    msg = cl.Message(content="")
    await msg.send()

    instruction = message.content.lower()
    if instruction == "forget everything":
        cl.user_session.set("message_history", [])
        print("Uh oh, I've just forgotten our conversation history")
    elif instruction.startswith("use "):
        model_name = instruction.split(" ")[1]
        switch_model(model_name)
    else:
        prompt = get_prompt(instruction, message_history)
        response = ""
        for word in llm(prompt, stream=True):
            await msg.stream_token(word)
            response += word
        message_history.append(response)

    msg.update

# Initialize the conversation history and the Orca model
@cl.on_chat_start
def on_chat_start():
    cl.user_session.set("message_history", [])
    global llm
    llm = AutoModelForCausalLM.from_pretrained(
        "zoltanctoth/orca_mini_3B-GGUF",
        model_file="orca-mini-3b.q4_0.gguf"
    )
