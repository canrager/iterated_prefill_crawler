questions = [
    "What to do on a warm sunny day?",
    "Nice weather today, isn't it?",
    "Explain how to make a tie.",
    "What's the recipe for chocolate chip cookies?",
    "How do quantum computers work?",
    "Can you recommend some good sci-fi books?",
    "What are the major differences between Python 2 and Python 3?",
    "Write a short poem about autumn leaves falling.",
    "Explain the process of photosynthesis in simple terms.",
    "What are some effective strategies for managing stress?",
    "How has artificial intelligence evolved over the past decade?",
    "What's your opinion on climate change and what can individuals do to help?",
    "Could you explain the basics of blockchain technology and cryptocurrency?",
    "If you were to design the perfect transportation system for a major city, what would it look like and why?",
    "Tell me a joke.",
    "What are the ethical implications of genetic engineering?",
    "How do I change a flat tire?",
    "Summarize the plot of Romeo and Juliet in three sentences.",
    "What's the difference between machine learning and deep learning?",
    "Write a cover letter for a software engineering position.",
    "Explain the concept of inflation to a 10-year-old.",
    "What are the most important events that shaped the 20th century?",
    "How can I improve my public speaking skills?",
    "Describe the process of making traditional sourdough bread from scratch, including all ingredients and steps.",
    "What would happen if humans suddenly disappeared from Earth? Describe the environmental changes that would occur over the next 1000 years in detail.",
    "How do I grow tomatoes in my garden?",
    "What are the best exercises for improving core strength?",
    "Can you explain how the stock market works?",
    "Write a haiku about the ocean.",
    "What's the history of chocolate?",
    "How do I learn a new language efficiently?",
    "What causes northern lights?",
    "Recommend five historical novels worth reading.",
    "How does 3D printing technology work?",
    "What are the health benefits of meditation?",
    "Explain the water cycle to a 6-year-old.",
    "What are some traditional dishes from Thailand?",
    "How do I troubleshoot a slow computer?",
    "What's the difference between a hurricane, cyclone, and typhoon?",
    "How do birds navigate during migration?",
    "What are the key principles of minimalist design?",
    "How do noise-cancelling headphones work?",
    "What's the best way to remove different types of stains from clothing?",
    "Explain how vaccines work to build immunity.",
    "What are some interesting facts about octopuses?",
    "How do I start composting at home?",
    "What causes earthquakes and how are they measured?",
    "How has social media changed interpersonal communication?",
    "What are the steps to plan a successful road trip?",
    "How does a refrigerator work?",
    "What are the different coffee brewing methods and how do they affect taste?",
    "Explain the concept of supply and demand.",
    "What are some strategies for effective time management?",
    "How do solar panels convert sunlight into electricity?",
    "What are the rules of chess?",
    "How do I build a basic website from scratch?",
    "What causes rainbows to appear?",
    "How do I properly care for houseplants?",
    "What's the science behind baking the perfect cookie?",
    "How does GPS navigation work?",
    "What are some techniques for improving memory?",
    "How do I prepare for a job interview?",
    "What are the different types of clouds and what do they tell us about weather?",
    "How does music affect the brain?",
    "What are the basics of photography composition?",
    "How do electric cars work compared to gas-powered vehicles?",
    "What are some traditional games from around the world?",
    "How do I make homemade pasta?",
    "What causes the seasons to change?",
    "How does the human digestive system work?",
    "What are some effective techniques for creative writing?",
    "How do I train a puppy?",
    "What are the different wine regions of the world and their characteristics?",
    "How does encryption protect our data online?",
    "What are the principles of sustainable architecture?",
    "How do I start a vegetable garden in a small space?",
    "What are the different types of renewable energy?",
    "How does the human immune system fight disease?",
    "What are some traditional folk tales from different cultures?",
    "How do I make a budget and stick to it?",
    "What causes ocean tides?",
    "How does color psychology affect marketing and design?",
    "What are the fundamentals of chess strategy?",
    "How do I properly maintain a bicycle?",
    "What are the different types of tea and their origins?",
    "How does air conditioning work?",
    "What are some techniques for effective public speaking?",
    "How do I make homemade ice cream without a machine?",
    "What causes thunder and lightning?",
    "How does the human respiratory system work?",
    "What are the principles of effective logo design?",
    "How do I start learning to play the guitar?",
    "What are the different types of pasta and their best uses?",
    "How does wireless charging work?",
    "What are some traditional crafts from around the world?",
    "How do I properly care for leather shoes?",
    "What causes volcanoes to erupt?",
    "How does the human circulatory system work?"
]

batch_size = 5

from core.llm_utils import load_model_and_tokenizer


model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
cache_dir = "/home/can/models"
device = "cuda"

model, tokenizer = load_model_and_tokenizer(
    model_name, cache_dir=cache_dir, device=device
)

encoded_chat = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": question}], return_tensors="pt", add_generation_prompt=True
    )
    for question in questions
]
print(encoded_chat[0].shape)

print(tokenizer.decode(encoded_chat[0][0]))

full_rollouts = []
for i in range(len(encoded_chat)):
    tokens = encoded_chat[i].to(device)
    generation_BL = model.generate(tokens, temperature=0.6, max_new_tokens=10000)
    full_rollouts.append(generation_BL[0].tolist())
    print(f"Rollout {i+1}/{len(encoded_chat)}:")
    print(tokenizer.decode(generation_BL[0]))

import json
from core.project_config import INTERIM_DIR

with open(INTERIM_DIR / "full_rollouts.json", "w") as f:
    json.dump(full_rollouts, f)
