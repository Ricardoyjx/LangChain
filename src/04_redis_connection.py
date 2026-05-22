import redis
import sys


def test_redis_connection():
    try:
        # -------------------------------------------------------
        # 配置区域
        # -------------------------------------------------------

        # 场景 A: 如果 Python 脚本运行在宿主机 (你的电脑上)
        # 且 Redis 是通过 docker run -p 6379:6379 启动的
        redis_host = "localhost"

        # 场景 B: 如果 Python 脚本也运行在一个 Docker 容器里
        # 且与 Redis 在同一个网络，或者使用的是 host 网络模式
        # redis_host = 'host.docker.internal' # Mac/Win 专用
        # redis_host = '172.17.0.1' # Linux 宿主机 IP 示例

        redis_port = 6379
        redis_password = None  # 如果设置了密码，请填写字符串，例如 'mypassword'

        # -------------------------------------------------------
        # 建立连接
        # -------------------------------------------------------

        # decode_responses=True 会自动将字节数据解码为字符串，方便打印
        client = redis.Redis(
            host=redis_host,
            port=redis_port,
            password=redis_password,
            decode_responses=True,
            socket_timeout=5,  # 设置5秒超时，防止一直卡住
        )

        # 发送 PING 命令
        response = client.ping()

        if response:
            print(f"✅ 连接成功! Redis 版本: {client.info('server')['redis_version']}")

            # 简单的读写测试
            client.set("py_test_key", "Hello from Python!")
            val = client.get("py_test_key")
            print(f"📝 读写测试成功: 获取到的值是 -> {val}")

    except redis.exceptions.ConnectionError as e:
        print(f"❌ 连接失败: 无法连接到 {redis_host}:{redis_port}")
        print(f"💡 错误详情: {e}")
        print("\n排查建议:")
        print("1. 检查 Docker 容器是否正在运行 (docker ps)")
        print("2. 检查端口映射是否正确 (-p 6379:6379)")
        print("3. 检查 Redis 是否开启了 protected-mode (需在配置文件中关闭)")

    except Exception as e:
        print(f"❌ 发生其他错误: {e}")


if __name__ == "__main__":
    test_redis_connection()
