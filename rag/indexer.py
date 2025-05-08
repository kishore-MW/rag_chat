import os
import json
import psycopg2
from psycopg2 import sql
from bedrock_inv.log import get_logger

logger = get_logger(__name__)

def get_postgres_connection():

    """
    Establish and return a connection to the PostgreSQL database.

    Returns:
        psycopg2 connection object.
    """

    # Database connection parameters
    db_params = {
        'dbname': os.getenv('DBNAME'),
        'user': os.getenv('DBUSER'),
        'password': os.getenv('PASSWORD'),
        'host': os.getenv('HOST'),
        'port': os.getenv('PORT')
    }
    try:
        conn = psycopg2.connect(**db_params)
        logger.info("Connected to PostgreSQL database successfully.")
        return conn
    except Exception as e:
        logger.warning(f"Error connecting to PostgreSQL: {e}")
        return None


def save_to_postgres(data_dict):
    """
    Save the extracted PDF data to the PostgreSQL database.

    Args:
        data_dict (dict): Dictionary containing page-level parsed data,
                          including doc_id, doc_name, text, page_number, and embedding.
    """
    table_name = "backup"
    conn = get_postgres_connection()

    if not conn:
        logger.error("No connection to PostgreSQL. Exiting.")
        return

    try:
        cursor = conn.cursor()
        insert_query = sql.SQL("""
            INSERT INTO {} (doc_id, doc_name, text, page_number, embedding)
            VALUES (%s, %s, %s, %s, %s)
        """).format(sql.Identifier(table_name))

        for page_number, content in data_dict.items():
            embedding = content.get('embedding')
            if embedding is not None:
                cursor.execute(insert_query, (
                    content['doc_id'],
                    content['doc_name'],
                    content.get('content') or content.get('text', ''),
                    content['page_number'],
                    json.dumps(embedding)
                ))
            else:
                logger.warning(f"Skipping page {page_number} due to missing embedding.")

        conn.commit()
        logger.info("All data successfully committed to PostgreSQL.")

    except Exception as e:
        print(f"Error saving data to PostgreSQL: {e}")
        logger.error(f"Error saving data to PostgreSQL: {e}", exc_info=True)

    finally:
        cursor.close()
        conn.close()


