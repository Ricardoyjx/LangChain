from src.generation.llm_client import create_ollama_client, verify_ollama_connection
from src.data_processing.parsers.factory import process_heterogeneous_data

MODEL_NAME = "qwen3.5:9b"
OLLAMA_BASE_URL = "http://localhost:11434"


def init_llm():
    """初始化 LLM，包含连接校验。"""
    print("正在初始化ollama本地大模型...")

    # 1. 先校验连接
    status = verify_ollama_connection(MODEL_NAME, OLLAMA_BASE_URL)
    if not status["ok"]:
        print(f"[错误] {status['message']}")
        return None
    if not status["model_ready"]:
        print(f"[警告] {status['message']}")
        return None

    # 2. 连接校验通过后，再创建客户端
    llm = create_ollama_client(MODEL_NAME, OLLAMA_BASE_URL, temperature=0.7)
    print(f"✓ ollama本地大模型初始化成功！{llm.name} 已准备就绪。")
    return llm


if __name__ == "__main__":
    # 初始化大模型
    llm = init_llm()
    # 如果模型初始化失败，直接退出
    if llm is None:
        print("无法继续执行，模型未准备好。")
        exit(1)

    files = [
        "./data/raw/美的2025年报.pdf",
    ]
    for file in files:
        result = process_heterogeneous_data(file)
        print(f"最终标准化结果 ({file}):")
        print(f"  内容长度: {len(result.get('content', ''))} 字符")
        print(f"  表格数量: {len(result.get('tables', []))}")
        print(f"  元数据: {result.get('metadata', {})}")
