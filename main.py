import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from chat import load_docs
from chat import process_query
from bedrock_inv.log import get_logger



logger = get_logger(__name__)

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    # Read the uploaded PDF
    try:
        pdf_bytes = await file.read()
        pdf_name = file.filename

        # Optionally save the file
        output_dir = "/home/maintwiz/Downloads/Document_retrival/pdf_manual"
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, pdf_name)
        with open(file_path, "wb") as f:
            f.write(pdf_bytes)

        # Example: Process the PDF (replace with your own function)
        load_docs()

        return JSONResponse(content={"message": "PDF processed successfully"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    
@app.post("/query")
def post_query(query: Query):
    logger.info(f"Received query: {query.question}")
    try:
        response = process_query(query.question)
        logger.info(f"Query processed successfully: {response}")
        return {"answer": response}
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
