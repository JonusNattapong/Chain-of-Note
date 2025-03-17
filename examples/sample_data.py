"""
Sample data generation for RAG examples.
"""
from typing import List, Dict

def get_sample_documents() -> List[Dict]:
    """Generate sample documents for testing.
    
    Returns:
        List of sample document dictionaries
    """
    documents = [
        {
            "content": """Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals. The term "artificial intelligence" had previously been used to describe machines that mimic and display "human" cognitive skills that are associated with the human mind, such as "learning" and "problem-solving". This definition has since been rejected by major AI researchers who now describe AI in terms of rationality and acting rationally, which does not limit how intelligence can be articulated.""",
            "metadata": {"source": "Wikipedia", "topic": "AI"}
        },
        {
            "content": """Machine learning (ML) is a field devoted to understanding and building methods that let machines "learn" – that is, methods that leverage data to improve computer performance on some set of tasks. Machine learning algorithms build a model based on sample data, known as training data, in order to make predictions or decisions without being explicitly programmed to do so. Machine learning algorithms are used in a wide variety of applications, such as in medicine, email filtering, speech recognition, agriculture, and computer vision, where it is difficult or unfeasible to develop conventional algorithms to perform the needed tasks.""",
            "metadata": {"source": "Wikipedia", "topic": "Machine Learning"}
        },
        {
            "content": """Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data. The goal is a computer capable of "understanding" the contents of documents, including the contextual nuances of the language within them. The technology can then accurately extract information and insights contained in the documents as well as categorize and organize the documents themselves.""",
            "metadata": {"source": "Wikipedia", "topic": "NLP"}
        },
        {
            "content": """Retrieval-Augmented Generation (RAG) is a technique in artificial intelligence that combines information retrieval with text generation. In RAG, a language model is augmented with a retrieval component that can fetch relevant documents or passages from a knowledge base. The retrieved information is then provided as additional context to the language model to generate more accurate, factual, and contextually relevant responses. RAG addresses the limitation of traditional language models which can generate plausible-sounding but incorrect information (hallucinations) by grounding the generation in verified external knowledge.""",
            "metadata": {"source": "AI Research Paper", "topic": "RAG"}
        },
        {
            "content": """Chain-of-Thought (CoT) prompting is a technique where a large language model is guided to break down complex reasoning tasks into intermediate steps. By explicitly showing the model how to reason through a problem step-by-step, CoT prompting improves performance on tasks requiring multi-step reasoning. Chain-of-Note is an evolution of CoT that focuses specifically on fact checking and source attribution, where the model first generates explicit notes about the retrieved information before formulating an answer.""",
            "metadata": {"source": "AI Research Paper", "topic": "Chain-of-Thought"}
        },
        {
            "content": """Hallucination in AI refers to when an AI system generates content that is factually incorrect or nonsensical but presented as if it were accurate. This is particularly problematic in generative models like large language models. Hallucinations occur when models generate plausible-sounding information that isn't grounded in their training data or provided context. Reducing hallucinations is a major focus of techniques like RAG (Retrieval-Augmented Generation) and Chain-of-Note, which ground the model's responses in verified information sources.""",
            "metadata": {"source": "AI Research Journal", "topic": "Hallucination"}
        }
    ]
    
    return documents

def get_sample_queries() -> List[Dict]:
    """Generate sample queries for testing.
    
    Returns:
        List of sample query dictionaries with expected topics
    """
    queries = [
        {"query": "What is artificial intelligence?", "expected_topics": ["AI"]},
        {"query": "How does machine learning work?", "expected_topics": ["Machine Learning"]},
        {"query": "Explain retrieval-augmented generation.", "expected_topics": ["RAG"]},
        {"query": "What is the Chain-of-Note technique?", "expected_topics": ["Chain-of-Thought"]},
        {"query": "How can AI hallucinations be reduced?", "expected_topics": ["Hallucination", "RAG"]},
    ]
    
    return queries
