# financial_rag_project 项目目录结构

financial_rag_project/
├── config/                     # 配置文件目录
│   ├── __init__.py
│   ├── settings.py             # 存放全局配置（如向量库地址、模型路径、阈值等）
│   └── logging_config.py       # 日志打印格式配置
│
├── data/                       # 数据存放目录（通常不提交到Git）
│   ├── raw/                    # 存放原始的 PDF、Word、Excel 等异构文件
│   ├── processed/              # 存放预处理后生成的标准格式 JSON 或 Parquet 文件
│   └── vector_db/              # 本地向量数据库的持久化存储目录（如 FAISS 索引文件）
│
├── src/                        # 核心源代码目录
│   ├── __init__.py
│   │
│   ├── data_processing/        # 【阶段一】数据处理模块
│   │   ├── __init__.py
│   │   ├── parsers/            # 异构文件解析器
│   │   │   ├── base.py         # 定义 BaseParser 抽象基类
│   │   │   ├── pdf_parser.py   # PDFParser 实现
│   │   │   ├── word_parser.py  # WordParser 实现
│   │   │   └── excel_parser.py # ExcelParser 实现
│   │   ├── chunker.py          # 文本切分与重叠处理逻辑
│   │   └── cleaner.py          # 文本清洗、去噪逻辑
│   │
│   ├── retrieval/              # 【阶段二】检索与后过滤模块
│   │   ├── __init__.py
│   │   ├── vector_store.py     # 向量数据库的增删改查封装
│   │   ├── keyword_store.py    # 关键词检索（如 Elasticsearch/BM25）封装
│   │   ├── hybrid_search.py    # 混合检索与 RRF 融合算法实现
│   │   ├── post_filter.py      # 后过滤链（去重、权限、时效性过滤）
│   │   └── reranker.py         # 精排模型（Rerank）的调用与打分逻辑
│   │
│   ├── generation/             # 【阶段三】结果生成模块
│   │   ├── __init__.py
│   │   ├── prompt_template.py  # 存放各类业务的 Prompt 模板
│   │   └── llm_client.py       # 大语言模型（LLM）的调用接口封装
│   │
│   ├── utils/                  # 公共工具模块
│   │   ├── __init__.py
│   │   ├── file_utils.py       # 文件路径处理、后缀获取等工具
│   │   └── embedding_client.py # Embedding 向量化模型的统一调用接口
│   │
│   ├── pipeline.py             # ⭐（新增）管线编排：串联 ingest / query 全流程
│   └── evaluator.py            # ⭐（新增）金融 RAG 评估器：数字准确率 + 忠实度
│
├── tests/                      # 单元测试目录（对应 src 里的模块进行测试）
│   ├── test_parsers.py
│   └── test_retrieval.py
│
├── logs/                       # 程序运行产生的日志文件目录
│
├── main.py                     # 项目主入口（比如提供 API 接口或命令行交互）
├── requirements.txt            # 项目依赖的 Python 库列表
└── README.md                   # 项目说明文档
