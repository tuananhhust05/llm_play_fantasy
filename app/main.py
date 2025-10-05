import os
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama

# 1. ĐỊNH NGHĨA MODELS (Pydantic)
# ==================================
class SynthesisRequest(BaseModel):
    articles: List[str]

class SynthesisResponse(BaseModel):
    summary: str


# 2. XỬ LÝ LOGIC VỚI LLM
# ========================

# Biến global để giữ model, chỉ tải một lần khi ứng dụng khởi động
llm = None

def load_model():
    """Tải mô hình LLM từ file GGUF."""
    global llm
    # Lấy đường dẫn model từ biến môi trường, nếu không có thì dùng mặc định
    model_path = os.getenv("MODEL_PATH", "/app/models/qwen1_5_1.5b-chat-q4_k_m.gguf")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy file model tại: {model_path}. Hãy đảm bảo bạn đã tải model vào thư mục 'models'.")

    print(f"Bắt đầu tải model từ: {model_path}...")
    llm = Llama(
        model_path=model_path,
        n_ctx=4096,      # Độ dài context tối đa
        n_gpu_layers=0,  # Chỉ chạy trên CPU
        n_threads=4,     # Số CPU core sử dụng
        verbose=True,
    )
    print("Tải model thành công!")

def synthesize_articles(articles: list[str]) -> str:
    """Tạo prompt, gửi đến LLM và trả về kết quả tổng hợp."""
    if not llm:
        raise RuntimeError("Model chưa được tải.")

    # Tạo prompt từ danh sách bài báo
    articles_text = ""
    for i, article in enumerate(articles, 1):
        articles_text += f"--- BÀI BÁO {i} ---\n{article}\n\n"

    prompt = f"""
<|im_start|>system
Bạn là một biên tập viên thể thao chuyên nghiệp. Nhiệm vụ của bạn là đọc và tổng hợp thông tin từ nhiều bài báo phân tích về cùng một trận đấu dưới đây để viết thành MỘT bài báo duy nhất, hoàn chỉnh, có cấu trúc rõ ràng và văn phong hấp dẫn, chuẩn cú pháp tiếng Việt. Chỉ sử dụng thông tin được cung cấp.
<|im_end|>
<|im_start|>user
Đây là các bài báo cần tổng hợp:

{articles_text}
Hãy viết một bài báo tổng hợp hoàn chỉnh từ những thông tin trên.
<|im_end|>
<|im_start|>assistant
"""
    
    # Gửi yêu cầu đến LLM
    print("Đang gửi yêu cầu tổng hợp đến LLM...")
    output = llm(
        prompt,
        max_tokens=1024,
        temperature=0.7,
        stop=["<|im_end|>"], # Dừng sinh văn bản khi gặp token này
        echo=False
    )
    
    print("LLM đã phản hồi.")
    return output['choices'][0]['text'].strip()


# 3. THIẾT LẬP API (FastAPI)
# ============================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code chạy khi server khởi động: Tải model vào RAM
    load_model()
    yield
    # Code chạy khi server tắt (nếu cần dọn dẹp)
    print("Server đang tắt.")

# Khởi tạo ứng dụng FastAPI
app = FastAPI(
    title="API Tổng Hợp Bài Báo",
    description="API sử dụng Qwen-1.5B để tổng hợp nhiều bài báo thành một.",
    lifespan=lifespan
)

@app.get("/", tags=["Health Check"])
def health_check():
    """Kiểm tra xem API có đang hoạt động không."""
    return {"status": "ok", "model_loaded": llm is not None}

@app.post("/synthesize", response_model=SynthesisResponse, tags=["Core"])
def synthesize_endpoint(request: SynthesisRequest):
    """Nhận danh sách các bài báo và trả về bài tổng hợp."""
    if not request.articles or not all(isinstance(a, str) for a in request.articles):
        raise HTTPException(status_code=400, detail="Dữ liệu đầu vào phải là một danh sách các chuỗi (bài báo).")

    try:
        summary = synthesize_articles(request.articles)
        return SynthesisResponse(summary=summary)
    except Exception as e:
        print(f"Lỗi xử lý: {e}")
        raise HTTPException(status_code=500, detail="Đã có lỗi xảy ra trong quá trình tổng hợp.")