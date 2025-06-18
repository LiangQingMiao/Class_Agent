# 安装依赖
# pip install langchain langchain-openai pydantic 

import os
from typing import List

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser


# --- 1. 定义我们期望的输出结构 (Define the Desired Output Structure) ---
# 使用 Pydantic 定义一个数据模型。
# 我们想要一个包含多个问题的列表。
# BaseModel 是 Pydantic 的基类。
# Field 用于提供额外的元数据，比如描述，这能帮助 LLM 更好地理解字段的含义。
class QuestionList(BaseModel):
    """一个用于存放从文本中提取的问题的模型"""

    questions: List[str] = Field(description="从文本中提取出的需要回答的问题列表")


model = ChatOpenAI(
    # 3. 指定 OpenRouter 支持的模型名称
    # 你可以在 https://openrouter.ai/models 找到所有可用模型
    # 很多模型都有免费版本，例如 "mistralai/mistral-7b-instruct:free"
    model="mistralai/mistral-7b-instruct:free",
    # 4. 指定 API 密钥 (它会自动从环境变量 OPENROUTER_API_KEY 或 OPENAI_API_KEY 读取)
    # 如果你没有设置环境变量，也可以在这里直接传入：
    api_key="",
    # 5. 指定 OpenRouter 的 API 基地址 (这是最重要的步骤！)
    base_url="https://openrouter.ai/api/v1",
    # temperature=0 依然适用，用于获得更确定的输出
    temperature=0,
)


# --- 第2步: 创建一个函数来实现问题提取 ---
def Find_Format(document_content: str) -> List[str]:
    """
    使用 LangChain 从给定的文档内容中提取所有问题。

    Args:
        document_content (str): 要分析的文本内容。

    Returns:
        List[str]: 从文本中提取出的问题列表。
    """
    try:
        # 初始化 LLM 模型
        # Temperature=0 表示我们希望模型的输出更具确定性和事实性，适合提取任务
        model = ChatOpenAI(
            # 3. 指定 OpenRouter 支持的模型名称
            # 你可以在 https://openrouter.ai/models 找到所有可用模型
            # 很多模型都有免费版本，例如 "mistralai/mistral-7b-instruct:free"
            model="mistralai/mistral-7b-instruct:free",
            # 4. 指定 API 密钥 (它会自动从环境变量 OPENROUTER_API_KEY 或 OPENAI_API_KEY 读取)
            # 如果你没有设置环境变量，也可以在这里直接传入：
            api_key="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",
            # 5. 指定 OpenRouter 的 API 基地址 (这是最重要的步骤！)
            base_url="https://openrouter.ai/api/v1",
            # temperature=0 依然适用，用于获得更确定的输出
            temperature=0,
        )

        # 初始化 Pydantic 输出解析器
        parser = PydanticOutputParser(pydantic_object=QuestionList)

        # 创建提示模板 (Prompt Template)
        # 模板中包含了对 LLM 的指令、格式化要求以及用户输入的占位符
        prompt_template = """
        您是一位精通文本分析的专家。您的任务是仔细阅读以下提供的文本内容，并从中提取出所有的【问题】。
        问题是指那些寻求信息、确认或建议的句子，通常以问号（? 或 ？结尾），但也可能不包含问号。
        请将找到的所有问题准确无误地提取出来。

        {format_instructions}

        请分析以下文本：
        ---
        {text_to_analyze}
        ---
        """

        prompt = ChatPromptTemplate.from_template(
            template=prompt_template,
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        # 构建 LangChain 链 (LCEL - LangChain Expression Language)
        # 这是一个将提示、模型和解析器串联起来的调用链
        chain = prompt | model | parser

        # 调用链并传入文本内容
        result = chain.invoke({"text_to_analyze": document_content})

        # 从解析结果中返回问题列表
        return result.questions

    except Exception as e:
        print(f"发生错误: {e}")
        return []


# --- 第3步: 示例用法 ---
if __name__ == "__main__":
    # 示例文本，包含中英文、带问号和不带问号的各种问题
    sample_text = """
    项目启动会议将于下周三上午10点举行。
    所有相关人员都必须参加。你明白了吗？
    会议的主要议题是讨论下一季度的预算分配。
    What is the primary goal for Q3?
    我们需要确定关键的绩效指标。请问谁负责市场分析部分？
    另外，我想知道项目的时间表达到了什么程度。
    The report is due on Friday. We need to work faster.
    Tell me who is in charge of the presentation slides.
    项目结束后，我们会有一个复盘会议。
    """

    print("--- 正在分析示例文本 ---")
    print(f"原始文本:\n{sample_text}")

    # 调用函数提取问题
    extracted_questions = Find_Format(sample_text)

    print("\n--- 提取完成 ---")
    if extracted_questions:
        print("提取到的问题列表 (list[str]):")
        # 打印结果的类型以验证
        print(f"输出类型: {type(extracted_questions)}")
        for i, q in enumerate(extracted_questions, 1):
            print(f"{i}. {q}")
    else:
        print("未在文本中找到任何问题。")
