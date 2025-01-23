# RAG Agent

## Overview
This project is a RAG AI agent built using LangGraph state graphs to manage the workflow of the agent. It routs questions to either a vectorstore or web search, retrieves relevant documents, and generates answers based on those documents.

## Features
- Routing questions to appropriate data sources.
- Retrieving documents from specified URLs.
- Generating answers based on retrieved documents.
- Grading the relevance and accuracy of generated answers.

## Installation
To install the required dependencies, run:
```
pip install -r requirements.txt
```

## Usage
To run the main script, execute:
```
python main.py
```