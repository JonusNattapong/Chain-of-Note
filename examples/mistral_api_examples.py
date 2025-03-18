# examples/mistral_api_examples.py
import os
from mistralai import Mistral, MistralException

# API Key Setup (Ensure the environment variable is set)
api_key = os.environ.get("MISTRAL_API_KEY")
if not api_key:
    raise ValueError("MISTRAL_API_KEY environment variable not set. Please set it before running this script.")

client = Mistral(api_key=api_key)

def demonstrate_chat_completion():
    """Demonstrates basic chat completion using the Mistral AI API."""
    model = "mistral-large-latest"
    try:
        chat_response = client.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "What is the best French cheese?",
                },
            ]
        )
        print("Chat Completion Response:")
        print(chat_response.choices[0].message.content)
    except MistralException as e:
        print(f"Error in chat completion: {e}")

def demonstrate_embeddings():
    """Demonstrates generating embeddings using the Mistral AI API."""
    try:
        embeddings_response = client.embeddings(
            model="mistral-embed",
            input=[
                "Embed this sentence.",
                "As well as this one."
            ]
        )
        print("\nEmbeddings Response:")
        print(embeddings_response.data)
    except MistralException as e:
        print(f"Error in embeddings: {e}")

def demonstrate_ocr():
    """Demonstrates OCR (Optical Character Recognition) using the Mistral AI API."""
    try:
        ocr_response = client.ocr(
            model="mistral-ocr-latest",  # Or another appropriate OCR model
            document={
                "type": "document_url",
                "document_url": "https://arxiv.org/pdf/2201.04234.pdf"  # Example PDF URL
            },
            include_image_base64=True #Optional
        )
        print("\nOCR Response:")
        print(ocr_response) # This will print the full OCR response, which can be quite large.
        # For a real application, you'd likely want to process and extract specific parts of the response.

    except MistralException as e:
        print(f"Error in OCR: {e}")

if __name__ == "__main__":
    print("Demonstrating Mistral AI API Examples...\n")
    demonstrate_chat_completion()
    demonstrate_embeddings()
    demonstrate_ocr()

    print("\nTo run this script, make sure you have the 'mistralai' package installed:")
    print("  pip install mistralai")
    print("And set the MISTRAL_API_KEY environment variable with your API key.")