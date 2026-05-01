# FastAPI - REST API Key
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, model_validator
from app.rag_pipeline import create_rag_pipeline
 
app = FastAPI()
 
qa_chain = create_rag_pipeline()
 
class Query(BaseModel):
    query: str | None = None
    question: str | None = None

    @model_validator(mode="after")
    def ensure_text(self):
        # Accept either `query` or `question` in request payload.
        if self.query is None and self.question is None:
            raise ValueError("Provide either 'query' or 'question' in request body")
        return self

    @property
    def text(self) -> str:
        return (self.query or self.question or "").strip()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={
            "error": "Invalid request body",
            "hint": "Use JSON like: {\"query\": \"your question\"} or {\"question\": \"your question\"}",
            "details": exc.errors(),
        },
    )
 
@app.get("/")
def home():
    return {"message": "RAG API Running"}
 
@app.post("/ask")
def ask(q : Query):
    result = qa_chain.invoke({"question": q.text})

    # LCEL chain currently returns a string answer.
    if isinstance(result, str):
        return {"response": result, "sources": []}

    response = result.get('answer', '')
    sources = result.get('source_documents', [])
    return {"response": response, "sources": sources}
 