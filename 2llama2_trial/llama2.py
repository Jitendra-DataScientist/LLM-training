from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7B-Chat-GGUF",
    model_file="llama-2-7b-chat.Q8_0.gguf",
    )


def get_prompt (instruction: str) -> str:
    system = "You are an AI assitant that gives helpful answers. You answer the questions in a short and consise manner."
    prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{instruction} [/INST]"
    print (prompt)
    return prompt


question = "Which city is the capital of India?"
for word in llm(get_prompt(question), stream=True):
    print (word, end="", flush=True)
