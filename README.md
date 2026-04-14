# psychology-liu: 自伤行为风险评估问卷相似度分析系统

本项目旨在利用大语言模型（LLM）模拟具有特定心理背景的“人格”（Personas），自动化回答标准化的自伤行为及相关心理评估问卷。通过收集不同模型、不同人格下的回答数据，进行后续的相似度分析、一致性评估以及结构方程模型（SEM）路径分析，探索 LLM 在模拟复杂心理状态时的表现与稳健性。

---

## 1. 项目核心逻辑
1. **人格模拟**：通过 `System Prompt` 为 LLM 设定特定的人生经历、性格特质和情绪倾向。
2. **自动化测验**：系统遍历所有配置的模型和问卷，以 JSON 格式捕获 LLM 的选择结果及背后的“内心感受（ai_reasoning）”。
3. **数据结构化**：结果持久化至 PostgreSQL，包含完整的对话历史和模型原始回复。
4. **统计分析**：对量表进行自动计分、效度/诚实性检查，并生成模型间的相似度指标（MAE, Cosine）和统计路径分析（SEM）。

---

## 2. 目录结构与模块说明

### 根目录文件
- `问卷1.md` 至 `问卷5.md`: 原始 Markdown 格式问卷，包括 CTQ（童年创伤）、ERQ（情绪调节）、BRIEF（执行功能）、NSSI（自伤行为频率）等。
- `GEMINI.md`: 原始项目指令与技术栈说明。
- `pyproject.toml` / `uv.lock`: `uv` 项目依赖与环境配置文件。
- `runner.lock`: 运行锁，防止多个 `persona_runner.py` 实例同时写入数据库。

### `src/` 核心代码
- **`persona_runner.py`**: **核心入口脚本**。
  - 定义了 `PERSONAS` 列表（人格描述）。
  - 控制“人格 -> 模型 -> 问卷 -> 题目”的嵌套循环。
  - 维护对话上下文（History），确保跨问卷的语义连贯。
  - 处理 JSON 提取与解析重试（`MAX_JSON_RETRIES`）。
- **`analyze_data.py`**: **数据分析中枢**。
  - `get_score_from_text`: 鲁棒地从 LLM 文本中提取分值。
  - `parse_ctq / parse_erq / parse_brief / parse_nssi`: 针对不同量表的计分逻辑与效度过滤。
  - `run_sem_analysis`: 使用 `semopy` 或 `pingouin` 执行路径分析。
  - `calculate_similarity`: 计算不同模型在各维度得分上的 MAE 和余弦相似度。
- **`visualize_results.py`**: **结果可视化**。
  - 调用分析逻辑并将结果绘制为美化后的 Seaborn 柱状图，保存至 `output/`。
- **`visualize_2.py`** (位于根目录): **高级可视化脚本**。
  - 提供了更专业、美学程度更高的图表排版，支持多维度堆叠对比和汇总图（`model_comparison_summary.png`）的生成。
- **`db.py`**: 数据库交互层。负责表结构初始化及 `ai_persona_runs` / `questionnaire_answers` 表的读写。
- **`questionnaire_parser.py`**: 问卷解析器，将 Markdown 文本转换为结构化对象。
- **`llm/client.py`**: 通用 LLM 客户端，封装了 OpenAI SDK，支持流式输出控制及 `<think>` 标签清理。
- **`config/llm_config.py`**: 配置中心，存放 API 密钥、并发模型列表及针对特定模型的参数适配。

---

## 3. 安装与运行指南

### 环境准备
本项目推荐使用 `uv` 进行包管理：
```bash
# 同步安装所有依赖
uv sync
```

### 数据库配置
1. 确保已安装 PostgreSQL。
2. 在 `src/db.py` 或环境变量中修改连接信息。
3. 执行数据库初始化脚本：
   ```bash
   python src/verify_db.py
   ```

### 运行流程
1. **配置模型**：在 `src/config/llm_config.py` 中填入你的 API 密钥并启用相关模型。
2. **启动数据采集**：
   ```bash
   python src/persona_runner.py
   ```
   该过程可能耗时较长（取决于题目数量和模型并发），请关注终端实时进度。
3. **数据分析与可视化**：
   ```bash
   # 查看统计分析报表（终端输出）
   python src/analyze_data.py
   
   # 生成基础对比图表
   python src/visualize_results.py
   
   # 生成高级美化版及汇总对比图表（推荐）
   python visualize_2.py
   ```
   图表将存放在 `output/` 目录下。

---

## 4. 数据输出说明
除了数据库存储和图片输出外，项目在分析过程中可能会生成以下文件：
- `所有问卷最终得分表.xlsx`: 汇总所有模型在所有维度上的评分数据。
- `CTQ详细得分表.xlsx`: 专门针对 CTQ 量表的各模型详细得分（含效度检查结果）。
- `output/*.png`: 模型间各心理维度的对比可视化结果。

---

## 5. 扩展指南（面向接手者）

### 如何添加新的人格？
在 `src/persona_runner.py` 的 `PERSONAS` 列表中增加一个字典：
```python
PERSONAS = [
    {
        "id": "new_persona",
        "description": "此处填入详细的人格设定描述..."
    }
]
```

### 如何添加新的问卷？
1. 在根目录创建新的 `问卷X.md`。
2. 确保格式与现有问卷一致（例如题号加粗或列表形式）。
3. **关键步骤**：在 `src/analyze_data.py` 中编写对应的 `parse_XXX` 计分函数，并在 `main` 中调用它。

### 关键注意事项
- **JSON 提取**：LLM 有时会返回非纯 JSON 的文本。`_parse_reply` 函数使用了正则提取。如果发现大量解析失败，请检查 `persona_runner.py` 中的纠错 Prompt 或增加重试次数。
- **反向计分**：在 `analyze_data.py` 中处理 CTQ 等量表时，请务必核对反向计分题目（如 CTQ 的情感忽略维度）。
- **无效回答**：CTQ 计分包含效度检查（如第10题），如果不符合逻辑，该次运行结果会被标记为无效并在分析中剔除。

---

## 5. 依赖项
- Python 3.12+
- `psycopg2-binary`: PostgreSQL 交互
- `pandas`, `numpy`: 数据处理
- `matplotlib`, `seaborn`: 可视化
- `semopy`, `pingouin`: 统计路径分析
- `openai`: LLM 调用接口
