# main.py
from src.state_graph.graph_setup import setup_graph
from src.retrieval.retriever import create_retriever
import os

def main():
    print("---INITIALIZE RETRIEVER---")
    try:
        
        # Initialize the retriever
        retriever = create_retriever("./config/data_config.json")

        # Use the retriever as needed
        print("Retriever initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize retriever: {e}")
    graph = setup_graph()
    # Save the image to a file
    image_path = 'graph_image.png'
    graph.get_graph().draw_mermaid_png(output_file_path=image_path)
    # Open the image
    os.system(f"code {image_path}")
    
if __name__ == "__main__":
    main()
