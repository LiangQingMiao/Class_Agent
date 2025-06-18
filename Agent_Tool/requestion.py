import asyncio
import nest_asyncio
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# 允许在已运行的事件循环中嵌套调用协程
nest_asyncio.apply()


class QwenModel:
    def __init__(
        self,
        api_key: str = "",
        model_name: str = "qwen3-235b-a22b",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ):
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_refined_question(self, context: str, question: str):
        prompt = f"""
你是一个信息整合助手，请根据以下内容优化用户的问题：
【原始问题】：{question}
【上下文】：
{context}

要求：
1. 语言简洁清晰；
2. 从上下文中提取关键词（如研究领域、技术术语、核心概念等）；
3. 新问题要更具体、更具针对性，并体现文件所述领域的专业术语；
4. 输出为一句话。
        """
        response = await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=512,
            extra_body={"enable_thinking": False}
        )
        return response.choices[0].message.content


async def refine_question_async(context: str, question: str):
    qwen = QwenModel()
    refined_question = await qwen.generate_refined_question(context, question)
    return refined_question


def refine_question(context: str, question: str):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 正在运行的 event loop，使用 ensure_future + run_until_complete
            task = asyncio.ensure_future(refine_question_async(context, question))
            return loop.run_until_complete(task)
        else:
            return loop.run_until_complete(refine_question_async(context, question))
    except Exception as e:
        print(f"⚠️ 调用失败：{str(e)}")
        return ""

# 示例入口
if __name__ == "__main__":
    sample_context = input("请输入你要输入的文本：")
    sample_question = input("请输入你要查询的问题：")

    refined = refine_question(sample_context, sample_question)
    print("Refined Question:", refined)
# user_question = input("请输入你要查询的问题：")