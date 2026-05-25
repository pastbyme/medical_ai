import asyncio
import json
import uuid
import os
import re
import warnings
import concurrent.futures
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from mcp import ClientSession
from mcp.client.sse import sse_client
from databases import Database
from datetime import datetime

# ---------- Transformers 相关 ----------
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, logging

# ---------- 关闭不必要的警告和日志（参照 exa_model.py）----------
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

# ---------- 配置 ----------
LOCAL_MODEL_PATH = r"D:\medical_aide\exported_merged_model"   # 本地模型目录
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- 全局变量 ----------
model = None
tokenizer = None
executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)  # 用于同步推理

MCP_SERVER_URL = "http://127.0.0.1:8001/sse"

MYSQL_USER, MYSQL_PASSWORD = "root", os.getenv("MYSQL_PASSWORD", "root123")
DATABASE_URL = f"mysql+aiomysql://{MYSQL_USER}:{MYSQL_PASSWORD}@localhost:3306/agenthistorl"
database = Database(DATABASE_URL)

_tools_cache: List[Dict[str, Any]] = []

# ---------- 数据模型 ----------
class ChatRequest(BaseModel):
    query: str
    sys_prompt: str = "你是一位专业的医疗助手，请提供准确、有用的健康信息。"
    history_len: int = 5
    history: List[Dict[str, str]] = []
    temperature: float = 0.3      # 参照 exa_model.py
    top_p: float = 0.8            # 参照 exa_model.py
    max_tokens: int = 2048        # 修改为 2048
    stream: bool = True
    conversation_id: Optional[str] = None

class NearbyRequest(BaseModel):
    lng: float
    lat: float

class EmailRequest(BaseModel):
    email: str
    advice_content: str

# ---------- 数据库操作 ----------
async def save_message(conv_id: str, role: str, content: str = None, tool_calls: List[Dict] = None,
                       tool_call_id: str = None):
    await database.execute(
        "INSERT INTO agentmessages (conversation_id, role, content, tool_calls, tool_call_id) VALUES (:conv_id, :role, :content, :tool_calls, :tool_call_id)",
        {"conv_id": conv_id, "role": role, "content": content,
         "tool_calls": json.dumps(tool_calls) if tool_calls else None, "tool_call_id": tool_call_id}
    )

# ---------- MCP 辅助 ----------
async def _call_mcp(func):
    try:
        async with sse_client(MCP_SERVER_URL) as streams, ClientSession(*streams) as session:
            await session.initialize()
            return await func(session)
    except Exception as e:
        print(f"[MCP] 失败: {e}")
        return None

async def fetch_mcp_tools():
    """获取MCP工具列表，如果MCP服务器不可用则返回空列表"""
    try:
        tools = await _call_mcp(lambda s: s.list_tools())
        if not tools:
            return []
        return [{"type": "function", "function": {"name": t.name, "description": t.description or "",
                                                  "parameters": t.inputSchema or {"type": "object", "properties": {}}}}
                for t in tools.tools]
    except Exception as e:
        print(f"[MCP] 获取工具失败: {e}")
        return []

async def call_mcp_tool(name: str, args: dict) -> str:
    """调用MCP工具，失败时返回错误信息"""
    try:
        res = await _call_mcp(lambda s: s.call_tool(name, arguments=args))
        texts = [c.text for c in res.content if hasattr(c, "text")] if res else []
        return "\n".join(texts) if texts else "工具执行成功"
    except Exception as e:
        return f"工具调用失败: {str(e)}"

# ---------- 工具调用解析 ----------
TOOL_CALL_PATTERN = re.compile(r'<tool_call>\s*({.*?})\s*</tool_call>', re.DOTALL)

def parse_tool_call(text: str) -> Optional[Dict]:
    """从模型输出中提取第一个 <tool_call> 标签内的 JSON"""
    match = TOOL_CALL_PATTERN.search(text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
    return None

def strip_tool_call(text: str) -> str:
    """移除文本中的 <tool_call> 标签"""
    return TOOL_CALL_PATTERN.sub('', text).strip()

# ---------- 核心对话（支持工具调用）----------
async def complete_with_tools(req: ChatRequest, conv_id: str):
    """
    使用本地 Transformers 模型完成对话。
    如果模型输出包含 <tool_call>，则自动调用 MCP 工具，并将结果再次送入模型，
    直到没有工具调用或达到最大迭代次数。
    """
    # 构建消息列表
    messages = []
    if req.sys_prompt:
        messages.append({"role": "system", "content": req.sys_prompt})

    # 注入工具描述（如果有工具可用）
    if _tools_cache:
        tools_desc = "你可以使用以下工具，调用格式为 <tool_call>{\"name\": \"工具名\", \"arguments\": {...}}</tool_call>\n"
        for tool in _tools_cache:
            func = tool["function"]
            tools_desc += f"- {func['name']}: {func['description']}\n  参数: {json.dumps(func['parameters'], ensure_ascii=False)}\n"
        # 将工具描述追加到 system prompt 中
        if messages and messages[0]["role"] == "system":
            messages[0]["content"] += "\n\n" + tools_desc
        else:
            messages.insert(0, {"role": "system", "content": tools_desc})

    # 添加历史消息（只保留 user/assistant，避免 tool 消息混淆模型格式）
    history_messages = req.history[-req.history_len:] if req.history_len else []
    for msg in history_messages:
        if msg["role"] in ["user", "assistant"]:
            messages.append({"role": msg["role"], "content": msg["content"]})

    # 添加当前用户消息
    messages.append({"role": "user", "content": req.query})
    await save_message(conv_id, "user", req.query)

    # 迭代：模型可能调用多次工具
    for iteration in range(5):  # 最多调用5次工具
        # 将消息列表转换为模型输入文本
        try:
            # 使用 chat_template 生成 prompt
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            print(f"[模板错误] {e}，使用简单拼接")
            # 降级方案：简单拼接
            prompt = ""
            for m in messages:
                if m["role"] == "system":
                    prompt += f"System: {m['content']}\n"
                elif m["role"] == "user":
                    prompt += f"User: {m['content']}\n"
                elif m["role"] == "assistant":
                    prompt += f"Assistant: {m['content']}\n"
            prompt += "Assistant: "

        # 异步执行模型推理（放到线程池）
        def _inference():
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    do_sample=True,
                    repetition_penalty=1.15,          # 参照 exa_model.py
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True
                )
            response_ids = outputs[0][inputs.input_ids.shape[1]:]
            response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
            return response_text

        loop = asyncio.get_event_loop()
        assistant_content = await loop.run_in_executor(executor, _inference)

        # 检查是否包含工具调用
        tool_call = parse_tool_call(assistant_content)
        if tool_call and "name" in tool_call and "arguments" in tool_call:
            # 记录助手消息（含工具调用标记）
            await save_message(conv_id, "assistant", assistant_content, [tool_call])
            messages.append({"role": "assistant", "content": assistant_content})

            # 调用工具
            tool_name = tool_call["name"]
            args = tool_call["arguments"]
            print(f"[工具调用] {tool_name}({args})")
            tool_result = await call_mcp_tool(tool_name, args)
            print(f"[工具结果] {tool_result[:100]}...")

            # 保存工具结果
            await save_message(conv_id, "tool", tool_result, tool_call_id=tool_call.get("id", ""))
            # 将工具结果加入到对话中（某些模型需要 tool 角色，但简单起见用 user 伪装）
            messages.append({
                "role": "user",
                "content": f"工具 {tool_name} 返回的结果是：\n{tool_result}\n请根据这个结果继续回答用户的问题。"
            })
            # 继续循环，让模型根据工具结果生成最终回复
            continue

        # 没有工具调用，正常输出
        final_content = strip_tool_call(assistant_content)  # 移除可能的残留标签
        await save_message(conv_id, "assistant", final_content)

        # 流式输出模拟（与原代码一致）
        if req.stream:
            async def generate():
                # 逐块返回（模拟流式）
                for i in range(0, len(final_content), 3):
                    yield final_content[i:i+3]
                    await asyncio.sleep(0.02)
            return final_content, generate()
        else:
            return final_content, None

    # 超过迭代次数
    error = "工具调用次数过多，已终止。"
    if req.stream:
        async def error_gen():
            yield error
        return error, error_gen()
    return error, None

# ---------- 生命周期 ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, _tools_cache
    # 连接数据库
    await database.connect()
    print("[数据库] 连接成功")
    try:
        await database.execute("""
            CREATE TABLE IF NOT EXISTS agentmessages (
                id INT PRIMARY KEY AUTO_INCREMENT,
                conversation_id VARCHAR(255),
                role VARCHAR(50),
                content TEXT,
                tool_calls JSON,
                tool_call_id VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        print("[数据库] 表结构检查完成")
    except Exception as e:
        print(f"[数据库] 建表警告: {e}")

    # 加载本地模型（使用 4-bit 量化配置，参照 exa_model.py）
    print(f"[模型] 正在从 {LOCAL_MODEL_PATH} 加载模型...")
    print("=" * 50)
    print("资源监控：目标 GPU 8GB (RTX 4070) | CPU ≤ 8GB")
    print("=" * 50)

    # 检查路径是否存在
    if not os.path.exists(LOCAL_MODEL_PATH):
        print(f"❌ 模型路径不存在: {LOCAL_MODEL_PATH}")
        raise RuntimeError("模型路径不存在")
    print(f"✅ 模型路径存在，目录中的文件: {os.listdir(LOCAL_MODEL_PATH)[:5]} ...")

    # 4-bit 量化配置
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # 加载 tokenizer（强制本地模式）
    tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_MODEL_PATH,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=True,   # 关键：禁止联网，只使用本地文件
    )
    tokenizer.pad_token = tokenizer.eos_token

    # 加载模型（量化 + 本地模式）
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_PATH,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,   # 同样强制本地加载
    )
    model.eval()

    # 显存占用验证
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024 ** 3
        reserved = torch.cuda.memory_reserved(0) / 1024 ** 3
        print(f"模型加载后 - 已分配显存: {allocated:.2f} GB | 预留显存: {reserved:.2f} GB")

    print(f"✅ 模型加载成功！设备: {model.device}")

    # 获取 MCP 工具列表
    print("[MCP] 正在获取工具列表...")
    _tools_cache = await fetch_mcp_tools()
    if _tools_cache:
        print(f"[MCP] 成功加载 {len(_tools_cache)} 个工具")
    else:
        print("[MCP] 未加载任何工具（MCP服务器可能未启动）")

    yield

    await database.disconnect()
    print("[数据库] 断开连接")

# ---------- FastAPI 应用 ----------
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- API 路由 ----------
@app.get("/")
async def root():
    return {
        "status": "healthy",
        "model": LOCAL_MODEL_PATH,
        "tools": [t["function"]["name"] for t in _tools_cache],
        "tools_loaded": len(_tools_cache) > 0,
        "mcp_connected": len(_tools_cache) > 0,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/status")
async def api_status():
    return {
        "status": "healthy",
        "model": LOCAL_MODEL_PATH,
        "tools_loaded": len(_tools_cache) > 0,
        "tools_count": len(_tools_cache)
    }

@app.post("/chat")
async def chat(req: ChatRequest):
    conv_id = req.conversation_id or str(uuid.uuid4())
    final, gen = await complete_with_tools(req, conv_id)
    if req.stream and gen:
        return StreamingResponse(gen, media_type="text/plain; charset=utf-8")
    return {"response": final, "conversation_id": conv_id}

@app.get("/conversations")
async def list_conversations(limit: int = 50, offset: int = 0):
    try:
        rows = await database.fetch_all(
            """SELECT conversation_id,
                      MAX(created_at) as last_active,
                      MIN(CASE WHEN role = 'user' THEN created_at END) as first_msg_time
               FROM agentmessages
               GROUP BY conversation_id
               ORDER BY last_active DESC LIMIT :limit OFFSET :offset""",
            {"limit": limit, "offset": offset}
        )
        result = []
        for row in rows:
            first_msg = await database.fetch_one(
                "SELECT content FROM agentmessages WHERE conversation_id = :cid AND role='user' ORDER BY created_at LIMIT 1",
                {"cid": row["conversation_id"]}
            )
            preview = first_msg["content"][:40] + "..." if first_msg and first_msg["content"] else "医疗咨询"
            msg_count = await database.fetch_one(
                "SELECT COUNT(*) as count FROM agentmessages WHERE conversation_id = :cid",
                {"cid": row["conversation_id"]}
            )
            result.append({
                "conversation_id": row["conversation_id"],
                "last_active": row["last_active"].isoformat() if row["last_active"] else None,
                "preview": preview,
                "message_count": msg_count["count"] if msg_count else 0
            })
        return result
    except Exception as e:
        print(f"[错误] 获取会话列表失败: {e}")
        return []

@app.get("/history/{conversation_id}")
async def get_history(conversation_id: str, limit: int = 100):
    try:
        rows = await database.fetch_all(
            "SELECT id, conversation_id, role, content, tool_calls, tool_call_id, created_at FROM agentmessages WHERE conversation_id = :cid ORDER BY created_at ASC LIMIT :limit",
            {"cid": conversation_id, "limit": limit}
        )
        return [{
            "id": r["id"],
            "conversation_id": r["conversation_id"],
            "role": r["role"],
            "content": r["content"] or "",
            "tool_calls": json.loads(r["tool_calls"]) if r["tool_calls"] else None,
            "tool_call_id": r["tool_call_id"],
            "created_at": r["created_at"].isoformat() if r["created_at"] else None
        } for r in rows]
    except Exception as e:
        print(f"[错误] 获取历史消息失败: {e}")
        return []

@app.delete("/conversation/{conversation_id}")
async def del_conversation(conversation_id: str):
    try:
        count = await database.fetch_one(
            "SELECT COUNT(*) as count FROM agentmessages WHERE conversation_id = :cid",
            {"cid": conversation_id}
        )
        await database.execute(
            "DELETE FROM agentmessages WHERE conversation_id = :cid",
            {"cid": conversation_id}
        )
        return {"status": "deleted", "conversation_id": conversation_id, "deleted_count": count["count"] if count else 0}
    except Exception as e:
        print(f"[错误] 删除会话失败: {e}")
        raise HTTPException(500, f"删除失败: {str(e)}")

@app.delete("/message/{message_id}")
async def del_message_and_following(message_id: int, conversation_id: str = Query(...)):
    try:
        target = await database.fetch_one(
            "SELECT id, created_at FROM agentmessages WHERE id = :id AND conversation_id = :cid",
            {"id": message_id, "cid": conversation_id}
        )
        if not target:
            raise HTTPException(404, "消息不存在")
        result = await database.execute(
            "DELETE FROM agentmessages WHERE conversation_id = :cid AND id >= :mid",
            {"cid": conversation_id, "mid": message_id}
        )
        return {"status": "deleted_following", "message_id": message_id, "deleted_count": result if isinstance(result, int) else 0}
    except HTTPException:
        raise
    except Exception as e:
        print(f"[错误] 删除消息失败: {e}")
        raise HTTPException(500, f"删除失败: {str(e)}")

@app.get("/health")
async def health_check():
    try:
        await database.fetch_one("SELECT 1")
        db_status = "connected"
    except:
        db_status = "disconnected"
    return {
        "status": "healthy",
        "database": db_status,
        "model_loaded": model is not None,
        "mcp": len(_tools_cache) > 0,
        "timestamp": datetime.now().isoformat()
    }

# ---------- 代理 MCP 工具的接口 ----------
@app.post("/api/nearby_hospitals")
async def proxy_nearby_hospitals(req: NearbyRequest):
    try:
        result = await call_mcp_tool("get_nearby_hospitals", {"lng": req.lng, "lat": req.lat})
        hospitals = json.loads(result) if isinstance(result, str) else []
        return {"hospitals": hospitals}
    except Exception as e:
        print(f"[代理] 附近医院查询失败: {e}")
        raise HTTPException(500, f"查询失败: {str(e)}")

@app.post("/api/send_advice")
async def proxy_send_advice(req: EmailRequest):
    try:
        result = await call_mcp_tool("send_medical_advice", {
            "email": req.email,
            "advice_content": req.advice_content
        })
        return {"message": result}
    except Exception as e:
        print(f"[代理] 发送邮件失败: {e}")
        raise HTTPException(500, f"发送失败: {str(e)}")

# ---------- 启动脚本 ----------
if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("🏥 医疗AI助手后端服务 (Transformers本地模型 + 4-bit量化)")
    print("=" * 60)
    print(f"📍 API文档: http://127.0.0.1:8000/docs")
    print(f"💬 聊天接口: http://127.0.0.1:8000/chat")
    print(f"🔧 状态检查: http://127.0.0.1:8000/api/status")
    print("=" * 60)
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )