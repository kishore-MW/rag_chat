import os
import shutil
from dotenv import load_dotenv
from bedrock_inv.aws_api import get_bedrock_client, get_logger, stream_vision_converse
from rag.indexer import save_to_postgres
from rag.retrivel import search_and_unpack_results
from rag.pdf_parse import docling_pdf_locally
from rag.history import read_last_responses

logger = get_logger(__name__)

load_dotenv()

client = get_bedrock_client()


def generate_response_with_text(query,doc_name=None):

    """
    Generate response using query.

    Args:
        query: the question what the user asks.

    Returns:
        Structure the query along with system prompt.
    """
    
    vec_results = search_and_unpack_results(query,doc_name)
    
    if vec_results:
        text = vec_results["text"][0]
        file_name = vec_results['doc_name'][0]
        image_path = vec_results['image_path']
        page_num = vec_results["page_number"][0]
        similarity = vec_results["similarity"]
        print("Similarity:", similarity)
        print("Page Number:", page_num) 
        print("Text:", text[:20])
        print("Image path:", image_path)
        print("doc_name:",file_name)   
    else:
        text = ""
        image_path = None
        file_name = ""

    last_query,last_response = read_last_responses(n=2)

    if text:
        
        user_msg = (
                "You are an expert AI assistant with buffer memory. You maintain awareness of only the last 5 user queries and your responses. "
                "Use these recent interactions to ensure contextual relevance and conversational continuity. "
                "Prioritize the latest exchanges and disregard any stale or unrelated information beyond these.\n"
                f"query_list:\n{last_query}\n"
                f"response_list:\n{last_response}\n")
        
        user_msg += (
                        f"""You are an expert assistant answering questions using both text and images.

                        Context:
                        - The input text includes `<!-- image -->` markers for images.
                        - A single attached collage contains all these images, labeled "Image 1", "Image 2", etc.
                        - Each `<!-- image -->` corresponds to its numbered image in order.

                        Instructions:
                        - Use the text and the labeled images as needed.
                        - Refer to image labels if the query mentions them.
                        - Be concise and accurate. Do not use external knowledge unless required.

                        Text:
                        {text}

                        Query: {query}
                        Answer:"""

        )

    else:
        
        user_msg = (
                "You are an expert AI assistant with buffer memory. You maintain awareness of only the last 5 user queries and your responses. "
                "Use these recent interactions to ensure contextual relevance and conversational continuity. "
                "Prioritize the latest exchanges and disregard any stale or unrelated information beyond these.\n"
                f"query_list:\n{last_query}\n"
                f"response_list:\n{last_response}\n"
                "You are a helpful and context-aware assistant. Use relevant past exchanges only if helpful.\n"
                "- Be concise and factual.\n"
                "- If the query is general, limit your response to 2 lines.\n"
                "- If clarification is needed, ask a short follow-up.\n"
                f"Query: {query}\nAnswer:"
            )

    return user_msg, image_path


pdf_folder = os.path.abspath(os.path.curdir) + "/pdf_manual"
output_folder = os.path.abspath(os.path.curdir) + "/text_and_image_chunks"
processed_folder = os.path.abspath(os.path.curdir) + "/processed_pdfs"

def load_docs():
    """
    Load and process all PDFs from the specified folder using SmallDocling.

    Processes each PDF only once, extracts text and images, saves results to the database,
    and moves processed files to a separate folder.
    """
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(processed_folder, exist_ok=True)

    processed_names = {os.path.splitext(f)[0] for f in os.listdir(processed_folder)}

    for file in pdf_files:
        try:
            pdf_name = os.path.splitext(file)[0]
            pdf_path = os.path.join(pdf_folder, file)

            if pdf_name in processed_names:
                logger.warning(f"PDF '{pdf_name}' already processed. Skipping.")
                continue

            logger.info(f"Parsing PDF: {file}")
            logger.info("Processing with Docling (this may take time)...")
            doc_output_folder = os.path.join(output_folder, pdf_name)
            os.makedirs(doc_output_folder, exist_ok=True)
            parsed_data = docling_pdf_locally(pdf_path, doc_output_folder)
            print(parsed_data)
            if parsed_data:
                save_to_postgres(parsed_data)
            else:
                logger.warning(f"No data returned for {file}.")

            shutil.move(pdf_path, os.path.join(processed_folder, file))
            logger.info(f"Moved processed PDF to {processed_folder}.")

        except Exception as e:
            logger.error(f"Error processing '{file}': {e}", exc_info=True)

            


# query = "What are the two versions of the Minicap level switch, and how do they differ?"

def process_query(query,doc_name):
    logger.info(f"Input query : {query}")
    user_msg, image = generate_response_with_text(query,doc_name)
    logger.info(f"System_prompt along with context: \n {user_msg}")
    try:
        if image:
            logger.info(f"image : {image[0][0]}")
    except: 
         image = None
    result = stream_vision_converse(client, user_msg, image, query)
    return result