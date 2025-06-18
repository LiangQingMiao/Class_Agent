import asyncio
import websockets
import json
import os
import base64
from datetime import datetime
from LLM import agent
from document_reader import DocumentReader
import time
from werkzeug.utils import secure_filename
import re

# 创建上传文件目录
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# 允许的文件类型（不区分大小写）
ALLOWED_EXTENSIONS = {'.png', '.doc', '.docx', '.pdf', '.txt'}


def clean_base64_data(data: str) -> str:
    """去除 base64 数据中的头部信息"""
    return re.sub(r'^data:.*?;base64,', '', data)

def is_allowed_file(filename):
    """检查文件类型是否允许"""
    ext = os.path.splitext(filename)[1].lower()
    print(f"\n[文件类型检查] 文件扩展名: {ext}")
    print(f"[文件类型检查] 允许的扩展名: {ALLOWED_EXTENSIONS}")
    return ext in ALLOWED_EXTENSIONS

async def save_file(file_data: str, filename: str, file_type: str) -> tuple[bool, str, str]:
    """保存上传的文件"""
    try:
        uploads_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
        os.makedirs(uploads_dir, exist_ok=True)
        
        # 生成安全的文件名
        safe_filename = secure_filename(filename)
        if not safe_filename:
            safe_filename = f"upload_{int(time.time())}"
        
        file_path = os.path.join(uploads_dir, safe_filename)

        # 清洗base64头部（关键步骤）
        try:
            cleaned_data = clean_base64_data(file_data)
            file_content = base64.b64decode(cleaned_data)
        except Exception as e:
            print(f"\n[错误] Base64解码失败: {str(e)}")
            return False, "文件数据解码失败", ""
        
        # 写入文件
        try:
            with open(file_path, 'wb') as f:
                f.write(file_content)
            print(f"\n[文件保存] 成功保存到: {file_path}")
            return True, safe_filename, file_path
        except Exception as e:
            print(f"\n[错误] 保存文件失败: {str(e)}")
            return False, f"保存文件失败: {str(e)}", ""
    except Exception as e:
        print(f"\n[错误] 处理文件时出错: {str(e)}")
        return False, f"处理文件时出错: {str(e)}", ""

def print_json_message(data):
    """打印格式化的JSON消息"""
    print("\n[收到的JSON消息]")
    print("=" * 50)
    if data.get('text'):
        print(f"文本内容: {data['text']}")
    print("\n文件信息:")
    print(f"  文件名: {data.get('filename', '未提供')}")
    print(f"  文件类型: {data.get('fileType', '未提供')}")
    print(f"  文件大小: {data.get('fileSize', '未提供')}")
    if data.get('fileData'):
        print("  文件数据: [base64编码数据]")
    else:
        print("  文件数据: 未提供")
    print("=" * 50)

async def websocket_handler(websocket):
    try:
        print("\n[连接] 新的客户端已连接")
        
        # 发送欢迎消息
        welcome_message = {
            'type': 'welcome',
            'content': '您好呀，我是艾森特，我可以帮你完成更新教材知识、理解整理文档、撰写教案、出题等工作！'
        }
        await websocket.send(json.dumps(welcome_message))
        print("\n[系统] 已发送欢迎消息")
        
        async for message in websocket:
            try:
                # 解析JSON消息
                data = json.loads(message)
                print_json_message(data)

                 # 处理文件上传
                if data.get('fileData') or data.get('filename'):
                    print(f"\n[文件上传] 开始处理文件: {data.get('filename', '未命名文件')}")
                    
                    success, result, filepath = await save_file(
                        data.get('fileData'),
                        data.get('filename'),
                        data.get('fileType')
                    )
                    
                    try:
                        if success:
                            response_data = {
                                'status': 'success',
                                'message': '文件上传成功',
                                'filename': result
                            }
                            print(f"\n[文件上传] 成功: {result}")
                        else:
                            response_data = {
                                'status': 'error',
                                'message': f'文件上传失败: {result}'
                            }
                            print(f"\n[文件上传] 失败: {result}")
                        
                        print("\n[发送响应]")
                        print("=" * 50)
                        print(json.dumps(response_data, ensure_ascii=False, indent=2))
                        print("=" * 50)
                        
                        # await websocket.send(json.dumps(response_data))
                        # print("\n[系统] 文件上传响应已发送")
                        
                    except Exception as e:
                        print(f"\n[错误] 发送响应时出错: {str(e)}")
                
                # 处理文本消息
                if data.get('text'):
                    print(f"\n[用户消息] {data['text']}")
                    print("\n[大模型] 开始流式响应...")
                    
                    
                    # 新增：收集所有chunk
                    full_response = ""

                    async for chunk in agent(data['text']):
                        response_data = {
                            'status': 'streaming',
                            'type': 'stream',
                            'content': chunk,
                            'replace': True  # 添加replace标志，表示需要替换之前的内容
                        }
                        if response_data['content'] != "":
                            await websocket.send(response_data["content"])
                        # 新增：拼接完整内容
                        full_response = chunk  # 直接覆盖，而不是追加

                    print("\n[系统] 流式响应完成")

                    # 新增：打印完整大模型回调
                    print("=" * 50)
                    print("[大模型] 完整回调内容：")
                    print(full_response)
                    print("=" * 50)
                
               
                
            except json.JSONDecodeError:
                print("\n[错误] 无效的JSON格式")
                continue
            except Exception as e:
                print(f"\n[错误] 处理消息时出错: {str(e)}")
                continue
                
    except websockets.exceptions.ConnectionClosed:
        print("\n[连接] 客户端已断开连接")
    except Exception as e:
        print(f"\n[错误] 发生异常: {str(e)}")

async def main():
    server = await websockets.serve(
        websocket_handler,
        "localhost",
        8765,
        ping_interval=None
    )
    print("\n[系统] WebSocket 服务器已启动")
    print(f"[系统] 监听地址: localhost:8765")
    print("[系统] 支持的文件类型: PNG, DOC, DOCX, PDF, TXT")
    print(f"[系统] 文件将保存在: {UPLOAD_DIR}")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main()) 