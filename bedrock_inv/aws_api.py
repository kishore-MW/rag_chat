import boto3
from dotenv import load_dotenv
from PIL import Image
import io
import os
import json
import ast
import traceback
from bedrock_inv.log import get_logger
from rag.history import history_write


logger = get_logger(__name__)

load_dotenv()

def load_and_resize_image(image_path, max_size=(1024, 1024)):
    """
    Load and resize image to be within the max_size pixel limits.
    Returns image bytes in JPEG format.
    """
    try:
        with Image.open(image_path) as img:
            img.thumbnail(max_size)  # Resize to fit within max_size
            byte_stream = io.BytesIO()
            img.convert("RGB").save(byte_stream, format="PNG")
            return byte_stream.getvalue()
    except Exception as e:
        logger.error(f"Failed to load or resize image: {e}")
        return None

def get_bedrock_client():
    """
    Initialize and return a Bedrock client.
    """
    try:
        AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
        AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
        AWS_REGION = os.getenv('AWS_REGION')
        if not AWS_ACCESS_KEY or not AWS_SECRET_KEY or not AWS_REGION:
            raise ValueError("AWS credentials are not set in environment variables.")
        client = boto3.client(
            'bedrock-runtime',
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name=AWS_REGION,
        )
        print("Bedrock client initialized successfully.")
        logger.info("Bedrock client initialized successfully.")
        return client
    
    except Exception as e:
        logger.error(f"Error initializing Bedrock client: {e}")
        print(f"Error initializing Bedrock client: {e}")
        return None


def invoke_embedding_model(client, prompt):
    """
    Invoke the embedding model with the given prompt.

    Args:
        client: Bedrock client object.
        prompt: The input text for the embedding model.

    Returns:
        The embedding result.
    """
    try:
        embedding_payload = {
            "inputText": prompt
        }
        embedding_response = client.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps(embedding_payload)
        )
        embedding_result = embedding_response['body'].read()
        embedding_result = ast.literal_eval(embedding_result.decode('utf-8'))
        return embedding_result['embedding']
    except Exception as e:
        # print(f"Error invoking embedding model: {e}")
        logger.error(f"Error invoking embedding model: {e}")
        return None


def stream_vision_converse(client, prompt, image_path, query):
    """
    Stream only the text content from Bedrock vision model using Converse API.

    Args:
        client: Bedrock runtime client.
        prompt (str): Prompt text.
        image_path (List[List[str]]): Nested list with image file paths (expects [[path]]).

    Yields:
        str: Text chunks from contentBlockDelta.
    """
    try:
        content = [{"text": prompt}]

        if image_path:
            path = image_path[0][0]
            image_bytes = load_and_resize_image(path)

            content.append({
                "image": {
                    "format": "png",
                    "source": {"bytes": image_bytes}
                }
            })

        messages = [{"role": "user", "content": content}]
        inference_config = {
            "maxTokens": 4096,
            "temperature": 0.9,
            "topP": 0.1
        }

        response = client.converse_stream(
            modelId="us.meta.llama3-2-90b-instruct-v1:0",
            messages=messages,
            inferenceConfig=inference_config
        )

        stream = response.get("stream")
        if not stream:
            yield "[ERROR] No stream found"
            return
        stream_final = ""
        for event in stream:
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"]["delta"]
                text = delta.get("text", "")
                if text:
                    yield text
                    stream_final += text
                    
            elif "metadata" in event:
                metadata = event["metadata"]
                usage = metadata.get("usage")
                if usage:
                    info = (
                        f"\nToken usage:\n"
                        f"  Input: {usage['inputTokens']}, "
                        f"Output: {usage['outputTokens']}, "
                        f"Total: {usage['totalTokens']}\n"
                    )
                    logger.debug(info)

        logger.info(f"Response : {stream_final}")
        history_write(query,stream_final)

    except Exception as e:
        yield f"vision model invoking error {str(e)}"


def invoke_generative_model(client, prompt):

    try:
        payload = {
            "prompt": prompt,
            "temperature": 0.9,
            "max_gen_len": 1024,
            "stop": ["|", "\n\n"]
        }

        response = client.invoke_model(
            modelId="arn:aws:bedrock:us-east-1:498398192936:inference-profile/us.meta.llama3-2-11b-instruct-v1:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload)
        )

        response_body = response['body'].read().decode('utf-8')
        data = json.loads(response_body)
        return data.get('generation') or data.get('outputs') or data

    except Exception as e:
        # print(f"Error invoking generative model: {e}")
        logger.error(f"Error invoking generative model: {e}")
        traceback.print_exc()
        return None

