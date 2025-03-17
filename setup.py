from setuptools import setup, find_packages

setup(
    name="chain_of_note_rag",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "sentence-transformers>=2.2.2",
        "faiss-cpu>=1.7.4",
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "langchain>=0.0.267",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
        "python-dotenv>=1.0.0",
    ],
    author="AI Developer",
    author_email="zombitx64@gmail.com",
    description="A RAG system using Chain-of-Note technique to reduce hallucinations",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/JonusNattapong/Chain-of-Note",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
