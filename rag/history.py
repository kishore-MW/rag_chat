import json
import os

history_file = "rag/history.json"

def history_write(new_query,new_response):
    
    # Load existing history or start fresh
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = {"query":[],"responses": []}
    else:
        history = {"query":[],"responses": []}

    # Append the new response
    history["responses"].append(new_response)
    history["query"].append(new_query)

    # Save back to file
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=4)


def read_last_responses(n=5):
    try:
        with open(history_file, 'r') as f:
            history = json.load(f)
            responses = history.get("responses", [])
            query = history.get("query",[])
            return query[-n:],responses[-n:]  # Get the last n responses
    except (FileNotFoundError, json.JSONDecodeError):
        return [],[]


