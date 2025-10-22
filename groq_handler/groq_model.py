from groq import Groq
import os

# Initialize Groq client
client = Groq(api_key=os.getenv("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"))

def generate_message(prompt: str) -> str:
    """
    Calls Groq LLaMA model and returns generated text.
    """
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "You are a helpful AI real estate assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
    )
    return response.choices[0].message.content  # ✅ fixed
