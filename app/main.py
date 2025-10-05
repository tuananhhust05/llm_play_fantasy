import os
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama

# 1. DEFINE MODELS (Pydantic)
# ==================================
class SynthesisRequest(BaseModel):
    articles: List[str]

class SynthesisResponse(BaseModel):
    summary: str


# 2. LLM LOGIC HANDLING
# ========================

# Global variable to hold the model, loaded only once on startup
llm = None

def load_model():
    """Loads the LLM model from a GGUF file."""
    global llm
    # Get model path from environment variable, use default if not set
    model_path = os.getenv("MODEL_PATH", "/app/models/qwen1.5-1.5b-chat.Q4_K_M.gguf")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}. Please ensure you have downloaded the model into the 'models' directory.")

    print(f"Starting to load model from: {model_path}...")
    llm = Llama(
        model_path=model_path,
        n_ctx=4096,      # Maximum context length
        n_gpu_layers=0,  # Run on CPU only
        n_threads=4,     # Number of CPU cores to use
        verbose=True,
    )
    print("Model loaded successfully!")

def synthesize_articles(articles: list[str]) -> str:
    """Creates a prompt, sends it to the LLM, and returns the synthesized result."""
    if not llm:
        raise RuntimeError("Model is not loaded.")

    # Create the combined text from the list of articles
    articles_text = ""
    for i, article in enumerate(articles, 1):
        articles_text += f"--- SOURCE ARTICLE {i} ---\n{article}\n\n"

    # Define the new, detailed English prompt
    prompt = f"""
<|im_start|>system
You are an expert sports editor and journalist. Your mission is to synthesize the key information, facts, and analyses from the multiple provided articles about the same sports match. Your goal is to craft a single, comprehensive, and impressive news article that is more detailed and better structured than any of the source articles.

CRITICAL INSTRUCTION: Your response MUST ONLY contain the content of the synthesized article. Do not include any preambles, headings, or conversational text like 'Here is the synthesized article:'. Your response must begin directly with the article's first sentence.
<|im_end|>
<|im_start|>user
Here are the source articles to synthesize:

{articles_text}
Now, write the complete, final article based on the information provided above.
<|im_end|>
<|im_start|>assistant
"""
    
    # Send the request to the LLM
    print("Sending synthesis request to LLM...")
    output = llm(
        prompt,
        max_tokens=1024,
        temperature=0.7,
        stop=["<|im_end|>"], # Stop generating text when this token is encountered
        echo=False
    )
    
    print("LLM has responded.")
    return output['choices'][0]['text'].strip()


# 3. SETUP API (FastAPI)
# ============================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run on server startup: Load the model into RAM
    load_model()
    yield
    # Code to run on server shutdown (for cleanup if needed)
    print("Server is shutting down.")

# Initialize FastAPI app
app = FastAPI(
    title="Article Synthesis API",
    description="An API using Qwen-1.5B to synthesize multiple articles into one.",
    lifespan=lifespan
)

@app.get("/", tags=["Health Check"])
def health_check():
    """Check if the API is running and the model is loaded."""
    return {"status": "ok", "model_loaded": llm is not None}

@app.post("/synthesize", response_model=SynthesisResponse, tags=["Core"])
def synthesize_endpoint(request: SynthesisRequest):
    """Receives a list of articles and returns a single synthesized article."""
    if not request.articles or not all(isinstance(a, str) for a in request.articles):
        raise HTTPException(status_code=400, detail="Input must be a list of strings (articles).")

    try:
        summary = synthesize_articles(request.articles)
        return SynthesisResponse(summary=summary)
    except Exception as e:
        print(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during the synthesis process.")
