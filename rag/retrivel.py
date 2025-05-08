from rag.indexer import get_postgres_connection
from bedrock_inv.aws_api import invoke_embedding_model, get_bedrock_client

client = get_bedrock_client()   


def search_vector_store(query_embedding, top_k=1, doc_names = None):
    """
    Search the vector store in PostgreSQL using cosine similarity.
    Optionally restricts to one or more document names.

    Args:
        query_embedding: List of floats representing the query embedding.
        top_k: Number of top results to retrieve.
        doc_names: List of document names to restrict the search to.

    Returns:
        List of rows matching the query.
    """
    conn = get_postgres_connection()
    if not conn:
        print("No connection to PostgreSQL. Exiting.")
        return []

    try:
        cursor = conn.cursor()

        if doc_names:
            # Prepare a WHERE IN (%s, %s, ...) clause dynamically
            placeholders = ','.join(['%s'] * len(doc_names))
            search_query = f"""
                SELECT doc_id, doc_name, text, image_path, page_number,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM backup
                WHERE doc_name IN ({placeholders})
                  AND 1 - (embedding <=> %s::vector) > 0.39
                ORDER BY similarity DESC
                LIMIT %s;
            """
            cursor.execute(search_query, [query_embedding] + doc_names + [query_embedding, top_k])
        else:
            # No document filter
            search_query = """
                SELECT doc_id, doc_name, text, image_path, page_number,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM backup
                WHERE 1 - (embedding <=> %s::vector) > 0.39
                ORDER BY similarity DESC
                LIMIT %s;
            """
            cursor.execute(search_query, (query_embedding, query_embedding, top_k))

        results = cursor.fetchall()
        return results

    except Exception as e:
        print(f"Error searching vector store: {e}")
        return []

    finally:
        cursor.close()
        conn.close()



def search_and_unpack_results(query,doc_name):

    query_embedding = invoke_embedding_model(client,query)
    """
    Search the vector store and unpack the results into separate lists.

    Args:
        query_embedding: List of floats representing the query embedding.
        top_k: Number of top results to retrieve.

    Returns:
        A dictionary containing unpacked results.
    """
    search_results = search_vector_store(query_embedding,doc_names=doc_name)
    # print(search_results)

    if search_results:
        unpacked_results = {
            "doc_id": [],
            "doc_name": [],
            "text": [],
            "image_path": [],
            "page_number": [],
            "similarity": []
        }

        for result in search_results:
            unpacked_results["doc_id"].append(result[0])
            unpacked_results["doc_name"].append(result[1])
            unpacked_results["text"].append(result[2])
            unpacked_results["image_path"].append(result[3])
            unpacked_results["page_number"].append(result[4])
            unpacked_results["similarity"].append(result[5])
    else:
        unpacked_results = None

    return unpacked_results

