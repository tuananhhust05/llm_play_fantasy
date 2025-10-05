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

llm = None

def load_model():
    """Loads the LLM model from a GGUF file."""
    global llm
    model_path = os.getenv("MODEL_PATH", "/app/models/qwen1.5-1.5b-chat.Q4_K_M.gguf")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}.")

    print(f"Starting to load model from: {model_path}...")
    llm = Llama(
        model_path=model_path,
        n_ctx=8192,  # Giữ nguyên context window lớn như đã yêu cầu
        n_gpu_layers=0,
        n_threads=4,
        verbose=True,
    )
    print("Model loaded successfully!")

# --- HÀM MỚI ĐỂ CHIA NHỎ VĂN BẢN ---
def chunk_text(text: str, chunk_size_chars: int):
    """
    Chia một văn bản dài thành các đoạn nhỏ hơn, ưu tiên ngắt ở cuối đoạn văn.
    """
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""

    for p in paragraphs:
        # Nếu thêm đoạn văn này vào chunk hiện tại sẽ vượt quá giới hạn
        if len(current_chunk) + len(p) + 2 > chunk_size_chars:
            if current_chunk:  # Thêm chunk hiện tại vào danh sách nếu nó không rỗng
                chunks.append(current_chunk)
            current_chunk = p
        else:
            # Nối đoạn văn vào chunk hiện tại
            if current_chunk:
                current_chunk += "\n\n" + p
            else:
                current_chunk = p
    
    if current_chunk:  # Thêm chunk cuối cùng còn lại vào danh sách
        chunks.append(current_chunk)
        
    return chunks
# --- KẾT THÚC HÀM MỚI ---


def extract_key_points_from_article(article: str) -> str:
    """
    Step 1: Processes a SINGLE article (or a chunk) to extract structured key points.
    """
    prompt = f"""
<|im_start|>system
You are a research assistant. Your task is to read the following sports article and extract the most important facts and talking points in a structured list format. Focus on key events, player performances, and tactical analysis.
<|im_end|>
<|im_start|>user
Here is the article:

{article}

Extract the key points from this article.
<|im_end|>
<|im_start|>assistant
"""
    print(f"Extracting key points from text block snippet: '{article[:100]}...'")
    output = llm(
        prompt,
        max_tokens=512,
        temperature=0.2,
        stop=["<|im_end|>"],
        echo=False
    )
    return output['choices'][0]['text'].strip()

def detailed_synthesis(articles: list[str]) -> str:
    """
    Takes the extracted key points and synthesizes the final, detailed article.
    """
    if not llm:
        raise RuntimeError("Model is not loaded.")

    # --- LOGIC CHIA NHỎ (CHUNKING) ĐƯỢC THÊM VÀO ĐÂY ---
    # Ước tính kích thước chunk an toàn, nhỏ hơn n_ctx một chút.
    # Giả sử 1 token ~ 4 ký tự, để an toàn ta dùng hệ số 3.
    CHUNK_SIZE_CHARS = llm.n_ctx() * 3 

    processed_texts = []
    for article in articles:
        if len(article) > CHUNK_SIZE_CHARS:
            print(f"Article is too long ({len(article)} chars), chunking it...")
            chunks = chunk_text(article, CHUNK_SIZE_CHARS)
            processed_texts.extend(chunks)
            print(f"Split into {len(chunks)} chunks.")
        else:
            # Nếu bài báo không quá dài, giữ nguyên nó
            processed_texts.append(article)
    # --- KẾT THÚC LOGIC CHIA NHỎ ---

    # Step 1: Map - Trích xuất ý chính từ mỗi bài báo HOẶC MỖI ĐOẠN NHỎ
    all_key_points = []
    for text_block in processed_texts: # <-- Dùng danh sách đã được xử lý
        points = extract_key_points_from_article(text_block)
        all_key_points.append(points)

    combined_points = "\n\n---\n\n".join(all_key_points)

    # Step 2: Reduce - Dùng các ý chính đã trích xuất để viết bài báo cuối cùng
    final_prompt = f"""
<|im_start|>system
You are an expert sports editor and journalist. Your mission is to use the following collection of key points, extracted from multiple sources, to write a single, comprehensive, and impressive news article about the match.

Your article must be well-structured with clear paragraphs, engaging, and more detailed than a simple summary. Weave the facts and analyses together into a compelling narrative.

CRITICAL INSTRUCTION: Your response MUST ONLY contain the content of the synthesized article. Do not include any preambles, headings, or conversational text. Your response must begin directly with the article's first sentence.
<|im_end|>
<|im_start|>user
Here are the key points to use for the article:

{combined_points}

Now, write the complete, final article based on this information.
<|im_end|>
<|im_start|>assistant
"""
    
    print("Sending final synthesis request to LLM with combined key points...")
    output = llm(
        final_prompt,
        max_tokens=2048,
        temperature=0.7,
        stop=["<|im_end|>"],
        echo=False
    )
    
    print("LLM has responded with the final article.")
    return output['choices'][0]['text'].strip()


# 3. SETUP API (FastAPI)
# ============================

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    print("Server is shutting down.")

app = FastAPI(
    title="Advanced Article Synthesis API",
    description="An API using a two-step process with chunking to synthesize detailed articles.",
    lifespan=lifespan
)

@app.get("/", tags=["Health Check"])
def health_check():
    return {"status": "ok", "model_loaded": llm is not None}

@app.post("/synthesize", response_model=SynthesisResponse, tags=["Core"])
def synthesize_endpoint(request: SynthesisRequest):
    """Receives articles and returns a single, detailed, synthesized article."""
    if not request.articles or not all(isinstance(a, str) for a in request.articles):
        raise HTTPException(status_code=400, detail="Input must be a list of strings.")

    try:
        summary = detailed_synthesis(request.articles)
        return SynthesisResponse(summary=summary)
    except Exception as e:
        print(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during synthesis.")

