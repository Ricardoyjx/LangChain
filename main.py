from src.data_processing.parsers.factory import process_heterogeneous_data
from src.generation.llm_client import create_ollama_client


def main():
    # 初始化ollama本地大模型
    print("正在初始化ollama本地大模型...")
    llm = create_ollama_client(
        model_name="qwen3.5:9b",
        base_url="http://localhost:11434",
        temperature=0.7,
    )
    # fix: 并没有验证是否启动了大模型
    if llm:
        print(f"ollama本地大模型初始化成功！\n{llm.name} 已准备就绪。")
    else:
        print("ollama本地大模型初始化失败！")


if __name__ == "__main__":
    # 初始化大模型
    main()

    files = [
        "./data/raw/美的2025年报.pdf",
    ]
    for file in files:
        result = process_heterogeneous_data(file)
        print(f"最终标准化结果 ({file}):")
        print(f"  内容长度: {len(result.get('content', ''))} 字符")
        print(f"  表格数量: {len(result.get('tables', []))}")
        print(f"  元数据: {result.get('metadata', {})}")
