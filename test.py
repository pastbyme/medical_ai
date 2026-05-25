
#
# import os
# from mcp.server.fastmcp import FastMCP
#
# # 在创建实例时指定 host 和 port
# mcp = FastMCP("FileSystem", host="127.0.0.1", port=8001)
#
# @mcp.tool()
# def get_desktop_files() -> list:
#     """获取当前用户的桌面文件列表"""
#     return os.listdir(os.path.expanduser("~/Desktop"))
#
# @mcp.tool()
# def calculator(a: float, b: float, operator: str) -> float:
#     if operator == '+':
#         return a + b
#     elif operator == '-':
#         return a - b
#     elif operator == '*':
#         return a * b
#     elif operator == '/':
#         return a / b
#     else:
#         raise ValueError("无效运算符")
#
# if __name__ == "__main__":
#     # 只需要指定 transport，不再传递 host/port
#     mcp.run(transport='sse')



import os
import json
import httpx
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from mcp.server.fastmcp import FastMCP

# ------------------- 配置（请替换为真实值）--------------------
AMAP_SERVER_KEY = "1234567456789"   # 您的高德服务端 Key
# 邮箱配置（用于真实发送邮件）
SMTP_SERVER = "smtp.qq.com"          # QQ邮箱 SMTP 服务器
SMTP_PORT = 587
SMTP_USER = "132456789@qq.com"         # 替换为您的邮箱
SMTP_PASSWORD = "123456789"          # 替换为您的 SMTP 授权码
# -----------------------------------------------------------

# 创建 FastMCP 服务器（不能改变结构）
mcp = FastMCP("MedicalAssistant", host="127.0.0.1", port=8001)

# ---------- 工具1：获取周边医院（真实调用高德 API）----------
@mcp.tool()
async def get_nearby_hospitals(lng: float, lat: float) -> str:
    """
    根据经纬度获取周边医院信息
    """
    url = "https://restapi.amap.com/v3/place/around"
    params = {
        "key": AMAP_SERVER_KEY,
        "location": f"{lng},{lat}",
        "keywords": "医院",
        "types": "医院",
        "radius": 5000,
        "output": "JSON"
    }
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params)
        data = resp.json()
        if data.get("status") != "1":
            return "获取医院失败"
        pois = data.get("pois", [])
        hospitals = []
        for p in pois[:15]:
            lng_lat = p["location"].split(",")
            hospitals.append({
                "id": p["id"],
                "name": p["name"],
                "address": p.get("address", ""),
                "lng": float(lng_lat[0]),
                "lat": float(lng_lat[1]),
                "distance": int(p.get("distance", 0))
            })
        return json.dumps(hospitals, ensure_ascii=False)

# ---------- 工具2：发送就医建议（真实邮件 + 模板生成）----------
@mcp.tool()
async def send_medical_advice(email: str, advice_content: str) -> str:
    """
    将 AI 生成的就医建议内容发送到指定邮箱。
    参数说明：
    - email: 收件人邮箱地址
    - advice_content: AI 模型生成的完整就医建议文本（纯文本）
    """
    # 将纯文本转换为 HTML（保留换行）
    html_content = advice_content.replace('\n', '<br>')
    full_html = f"""
    <html>
    <body style="font-family: Arial, sans-serif;">
    <h2 style="color: #2c7da0;">🏥 医疗AI助手就医建议</h2>
    <div style="font-size: 14px; line-height: 1.6;">{html_content}</div>
    <hr>
    <p style="color: gray; font-size: 12px;">本建议由AI生成，仅供参考。如有紧急情况请立即前往医院。</p>
    </body>
    </html>
    """
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = "医疗AI助手就医建议"
        msg["From"] = SMTP_USER
        msg["To"] = email
        part = MIMEText(full_html, "html", "utf-8")
        msg.attach(part)
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_USER, email, msg.as_string())
        return f"✅ 就医建议已成功发送至 {email}"
    except Exception as e:
        return f"❌ 邮件发送失败: {str(e)}"


# ---------- 启动服务器 ----------
if __name__ == "__main__":
    # 使用 SSE 传输，不能改变
    mcp.run(transport='sse')

