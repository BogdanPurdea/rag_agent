# main.py
from src.state_graph.graph_setup import setup_graph
import os
from rich import print
from dotenv import load_dotenv

def save_graph_as_image(graph):
    # Save the image to a file
    image_path = 'graph_image.png'
    graph.get_graph().draw_mermaid_png(output_file_path=image_path)
    # Open the image
    os.system(f"code {image_path}")
    
def main():
    load_dotenv()
    graph = setup_graph()
    inputs = {
        "question": "What are the types of agent memory?", 
        "max_retries": 3
    }
    # inputs = {
    #     "question": "What are the models released today for llama3.2?",
    #     "max_retries": 3,
    # }
    for event in graph.stream(inputs, stream_mode="values"):
        llm_output = event.get("generation", "Nothing was generated.")
        print("Generation result:", llm_output)

if __name__ == "__main__":
    main()
