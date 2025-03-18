import os
from mistralai import Mistral

# --- API Key Setup Instructions ---
# Before running this code, you need to set the MISTRAL_API_KEY environment variable.
# You can obtain an API key from the Mistral AI platform (https://mistral.ai/).
#
# On Windows (Command Prompt):
# set MISTRAL_API_KEY=your_api_key
#
# On Windows (PowerShell):
# $env:MISTRAL_API_KEY = "your_api_key"
#
# On macOS/Linux:
# export MISTRAL_API_KEY=your_api_key
#
# Replace "your_api_key" with your actual Mistral API key.
# --- End API Key Setup Instructions ---


def get_embeddings(texts):
    """
    Generates embeddings for a list of texts using the Mistral AI API.

    Args:
        texts: A list of strings to embed.

    Returns:
        A list of embedding vectors, or None if an error occurs.
    """
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print("Error: MISTRAL_API_KEY environment variable not set.")
        return None

    client = Mistral(api_key=api_key)
    try:
        embeddings_response = client.embeddings(
            model="mistral-embed",
            input=texts
        )
        return embeddings_response.data
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
    
    def process_ocr(document_url, include_image_base64=True):
        """
    Performs OCR on a document using the Mistral AI API.

    Args:
        document_url: The URL of the document to process.
        include_image_base64: Whether to include base64 encoded images in the response.

    Returns:
        The OCR response object, or None if an error occurs.
    """
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print("Error: MISTRAL_API_KEY environment variable not set.")
        return None

    client = Mistral(api_key=api_key)
    try:
        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": document_url,
            },
            include_image_base64=include_image_base64,
        )
        return ocr_response
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def get_agent_completion(messages, max_tokens=100, model="mistral-large-latest"):
    """
    Gets a completion from a Mistral AI agent.

    Args:
        messages: A list of message objects representing the conversation.
        max_tokens: The maximum number of tokens to generate.
        model: The Mistral AI model to use.

    Returns:
        The agent's response, or None if an error occurs.
    """
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print("Error: MISTRAL_API_KEY environment variable not set.")
        return None

    client = Mistral(api_key=api_key)
    try:
        chat_response = client.chat.complete(
            model=model,
            messages=messages,
            max_tokens=max_tokens
        )
        return chat_response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


if __name__ == '__main__':
    # Example Usage (Requires MISTRAL_API_KEY to be set)
    print("--- Example Usage ---")
    print("1. Embeddings:")
    texts = ["Embed this sentence.", "As well as this one."]
    embeddings = get_embeddings(texts)
    if embeddings:
        print(f"Embeddings: {embeddings}")

    print("\\n2. OCR:")
    document_url = "https://arxiv.org/pdf/2201.04234"  # Example PDF URL
    ocr_result = process_ocr(document_url)
    if ocr_result:
        print(f"OCR Result: {ocr_result}")

    print("\\n3. Agent Completion:")
    messages = [
        {"role": "user", "content": "What is the best French cheese?"},
    ]
    completion = get_agent_completion(messages)
    if completion:
        print(f"Agent Completion: {completion}")