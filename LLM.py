import os
import shutil
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.llms import Tongyi
from typing import AsyncGenerator
from langchain.agents import AgentExecutor
from langchain.agents import create_react_agent
from langchain_core.tools import BaseTool
import aiofiles
import asyncio
from langchain import hub
from document_reader import DocumentReader
from Agent_Tool.Internet_search import  get_qwen_answer_async,get_qwen_answer
from Agent_Tool.requestion import refine_question
from Agent_Tool.StripMarkdown import strip_md
from Agent_Tool.find_format import Find_Format
from Agent_Tool.generate_ppt_text import generate_ppt_content
from Agent_Tool.rewriter import rewrite_questions
from Agent_Tool.Summary import summarize_text

# 这里需要将 DASHSCOPE_API_KEY 替换为你在阿里云控制台开通的 API KEY
os.environ["DASHSCOPE_API_KEY"] = ""

# 可以通过 model 指定模型
llm = ChatTongyi(model='qwen-max-2024-09-19')

async def agent(message: str) -> AsyncGenerator[str, None]:
    """流式返回agent的响应"""
    print("\n[开始处理消息]")
    print("=" * 50)
    print(f"用户消息: {message}")
    print("=" * 50)
    file_content = "" 
    # 检查uploads文件夹下的文件
    uploads_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
    if os.path.exists(uploads_dir):
        reader = DocumentReader()
        # 遍历uploads文件夹下的所有文件
        for filename in os.listdir(uploads_dir):
            file_path = os.path.join(uploads_dir, filename)
            if os.path.isfile(file_path):
                try:
                    file_content = reader.read_document(file_path)
                    if file_content is not None:
                        print(f"\n[文件内容] {filename}")
                        print("-" * 50)
                        print(file_content)
                        print("-" * 50)
                        
                        # 读取完文件内容后立即删除文件
                        try:
                            os.remove(file_path)
                            print(f"[已删除文件] {filename}")
                        except Exception as e:
                            print(f"[警告] 删除文件 {filename} 时出错: {str(e)}")
                except Exception as e:
                    print(f"\n[警告] 读取文件 {filename} 时出错: {str(e)}")
        
    else:
        print("\n[提示] uploads文件夹不存在或为空")

    System_prompt = (
    """    
    # # Name: 艾森特
    # # Role: 高级课程助教
  
    # # Profile:
    - Author: Liang
    - Version: 1.0
    - Language: 中文
    - Description: 高级课程助教是一个问题解决角色,旨在针对用户的问题规划解决方案
  
    ### Skill:
    1. 擅长理解任务
    2. 根据任务需要规划出需要的动作
  
    ## Rules:
    1.可以使用的动作列表有:【Search(搜索);Write_Markdown(更新改写PPT);Make_Question(制造问题);Format(查找文档中的格式);Strip_MD;Rewrite(完成普通撰写工作);Summary(对文件内容做总结，提取关键主题句)】
    2.要以完成用户目标为导向
    3.除了要执行的动作意外，其他汉字性质的思考过程不要出现

    ## Workflow:
    1.用户提出文字需求，有可能会上传相关文件
    2.根据用户文字需求，规划出所需要的动作列表

    ## Initialization:
    作为角色 "高级课程助教"，我严格遵守上述规则，使用中文与用户对话。

    ###一些案例："""
    )
    async with aiofiles.open("demo\demo.txt", "r", encoding="utf-8") as f:
        demo = await f.read()

    Prompt = (
        System_prompt
        + demo
        + "## Message:"
        + message
        + "\n## Ans:\n"
    )
    

    full_response = ""
    target_words = []  # 用于存储提取的特定单词

    # 使用一个变量来存储当前响应
    current_response = ""

    async for chunk in llm.astream(Prompt):
        if chunk.content:
            full_response += chunk.content
            current_response = chunk.content  # 更新当前响应
            # yield current_response  # 只yield当前chunk

    # 从完整响应中提取目标单词（关键代码）[1,3](@ref)
    import re

    # 编译正则表达式（忽略大小写，支持目标单词列表）
    target_words = ["Search", "Write_MarkDown", "Make_Question", "Format", "Write_Word", "Summary","Strip_MD","Rewrite"]
    pattern = re.compile(r'\b(' + '|'.join(target_words) + r')\b', re.IGNORECASE)

    # 测试匹配
    # full_response = "First, Search the data; then Write_Markdown for summary."
    found_words = pattern.findall(full_response)  # 输出: ['Search', 'Write_Markdown']

    # 去重并保留顺序[2,5](@ref)
    unique_words = []
    for word in found_words:
        if word not in unique_words:
            unique_words.append(word)

    print("全局任务规划:", unique_words)  # 输出: ['Summary', 'Search', 'Write_Word']
    # 【Search(搜索);Write_Markdown(更新改写PPT);Find_Format(查找格式);Wirte_Word(完成普通撰写工作);Summary(对文件内容做总结，提取关键主题句)】
    
    question = message
    search_answer = ""
    ans = ""
    question_list = []
    for word in unique_words:
        if word == "Search":
            try:
                search_answer = await get_qwen_answer_async(question)
                print(f"📢 收到回答：{search_answer}")
                ans = search_answer
            except Exception as e:
                print(f"⚠️ 调用失败：{str(e)}")
        elif word == "Make_Question":
            try:
                question = refine_question(file_content,question)
                print(f"📢 想要搜索：{question}")
            except Exception as e:
                print(f"⚠️ 调用失败：{str(e)}")
        elif word == "Strip_MD":
            try:
                print(ans)
                ans = strip_md(ans)
            except Exception as e:
                print(f"⚠️ strip_md调用失败：{str(e)}")
        elif word == "Rewrite":
            try:
                result, ans = rewrite_questions(question_list)
            except Exception as e:
                print(f"⚠️ 调用失败：{str(e)}")
        elif word ==  "Format":
            try:
                question_list = Find_Format(file_content)
                print(f"📢 需要撰写的内容如下：{question_list}")
            except Exception as e:
                print(f"⚠️ 调用失败：{str(e)}")
        elif word == "Write_MarkDown":
            try:
                ans = generate_ppt_content(ans)
                print(f"📢 ：正在保存MarkDown格式内容")
            except Exception as e:
                print(f"⚠️ 调用失败：{str(e)}")
        elif word == "Summary":
            try:
                ans = summarize_text(ans)
                print(f"📢：总结内容如下：{ans}")
            except Exception as e:
                print(f"⚠️ 调用失败：{str(e)}")    
    yield ans  # 直接yield answer，覆盖之前的"正在搜索....."
# ================================================================================
# # 分步规划
# async def agent(message: str, history: str = "", have_content: str = "") -> AsyncGenerator[str, None]:
#     """流式返回agent的响应"""
#     Skill_List = ("""  form action import Search,Understand,Wirte,think""")
#     System_prompt = ("""
#         你只能使用从action中导入的四个动作，每次只能输出一个最合适的动作，不要有任何多余的输出
#         There are some examples：\n""")
#     async with aiofiles.open("demo.txt", "r", encoding="utf-8") as f:
#         demo = await f.read()

#     Prompt = (
#         Skill_List
#         + System_prompt
#         + demo
#         + "## Message:"
#         + message
#         + "\n## Action:\n"
#         + history 
#         + ">"
#     )
    
#     # 直接使用提示词字符串
#     print(Prompt)

#     # 流式调用
#     full_response = ""
#     async for chunk in llm.astream(Prompt):
#         if chunk.content:
#             full_response += chunk.content
#             # 注释掉流式回调
#             yield chunk.content
#             # print(f"[大模型] 回调内容: {chunk.content}")

#     if full_response == "end":
#         return
#     ## 异步调用动作
#     # history = history + ">" + full_response + "\n" + "OK"+"\n"
#     # print("00")
#     # agent(message=message, history=history, have_content=have_content)



# =================================================================================
# # LangChain原生规划
# model = Tongyi()
# model.model_name = 'qwen-max-2024-09-19'

# tools = [
#     QwenSearchTool(),
#     DocumentReaderTool()
# ]

# async def agent(message: str, history: str = "", have_content: str = "") -> AsyncGenerator[str, None]:
#     prompt = hub.pull("hwchase17/react")
#     agent = create_react_agent(model, tools, prompt)
#     agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
#     result = agent_executor.invoke({'input': message})
#     print(result['output'])
