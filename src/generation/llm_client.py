import requests
from langchain_ollama import ChatOllama
from requests.exceptions import ConnectionError, Timeout


def create_ollama_client(model_name, base_url, temperature):
    return ChatOllama(
        name=model_name,
        model=model_name,
        base_url=base_url,
        temperature=temperature,
    )


def verify_ollama_connection(model_name: str, base_url: str, timeout: int = 5) -> dict:
    """验证 Ollama 服务是否正常运行且目标模型已就绪。

    Args:
        model_name: 模型名称（如 qwen3.5:9b）。
        base_url: Ollama 服务地址（如 http://localhost:11434）。
        timeout: 请求超时时间（秒）。

    Returns:
        dict: {"ok": bool, "message": str, "model_ready": bool}。
    """
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=timeout)
        if resp.status_code != 200:
            return {
                "ok": False,
                "message": f"Ollama 服务响应异常 (HTTP {resp.status_code})",
                "model_ready": False,
            }
        models = resp.json().get("models", [])
        model_names = [m["name"] for m in models]
        if model_name in model_names:
            return {
                "ok": True,
                "message": f"Ollama 服务正常，模型 {model_name} 已就绪",
                "model_ready": True,
            }
        else:
            return {
                "ok": True,
                "message": f"Ollama 服务正常，但模型 {model_name} 未找到，请先执行: ollama pull {model_name}",
                "model_ready": False,
            }
    except ConnectionError:
        return {
            "ok": False,
            "message": f"无法连接到 Ollama 服务 ({base_url})，请确认服务已启动",
            "model_ready": False,
        }
    except Timeout:
        return {
            "ok": False,
            "message": f"连接 Ollama 服务超时 ({base_url})",
            "model_ready": False,
        }
    except Exception as e:
        return {
            "ok": False,
            "message": f"连接 Ollama 服务时发生未知错误: {e}",
            "model_ready": False,
        }
