from transformers import pipeline, AutoTokenizer
from dotenv import load_dotenv

load_dotenv()

generator = pipeline('text-generation', model='google/gemma-2-2b-it', device_map='auto')
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

user_input = input("USER: ")
chat_history = [
  {"role": "user", "content": user_input},
]
while user_input != "exit":
  gemma_response = generator(chat_history, max_new_tokens=150, num_return_sequences=1, truncation=True)[0]['generated_text'][-1]['content']
  print("GEMMA: ", gemma_response)
  chat_history.append({"role": "assistant", "content": gemma_response})
  user_input = input("USER: ")
  chat_history.append({"role": "user", "content": user_input})