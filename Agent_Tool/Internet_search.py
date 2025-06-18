import asyncio
import aiohttp
from bs4 import BeautifulSoup
import re
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import sys
from asyncio import WindowsSelectorEventLoopPolicy

# ======================= Qwen 模型封装类 =======================
class QwenModelConfig:
    def __init__(
        self,
        api_key: str = "",
        model_name: str = "qwen3-235b-a22b",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ):
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_summary(self, context: str, question: str):
        prompt = f"""
你是一个信息整合助手，请根据以下内容回答用户的问题：
【问题】：{question}
【上下文】：
{context}

要求：
1. 尽可能丰富且详细
2. 回答与问题高度相关；
3. 不要添加额外信息；
4. 输出为一段话。
        """
        response = await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=512,
            extra_body={
                "enable_thinking": False
            }
        )
        return response.choices[0].message.content


# ======================= 百度搜索函数 =======================
async def search_internet(query: str, session: aiohttp.ClientSession):
    url = f"https://www.baidu.com/s?wd={query}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0 Safari/537.36"
    }

    async with session.get(url, headers=headers) as response:
        if response.status != 200:
            raise Exception(f"网络请求失败，状态码: {response.status}")

        content = await response.text()
        soup = BeautifulSoup(content, 'html.parser')
        results = []

        for item in soup.select('div.result'):
            title_tag = item.find('h3').find('a') if item.find('h3') else None
            link = title_tag['href'] if title_tag and 'href' in title_tag.attrs else ''
            title = title_tag.get_text(strip=True) if title_tag else ''

            results.append({
                'title': title,
                'link': link,
            })

        return results


# ======================= 网页内容提取 =======================
async def fetch_page_content(url: str, session: aiohttp.ClientSession):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0 Safari/537.36"
    }
    try:
        async with session.get(url, headers=headers, ssl=False) as response:
            if response.status != 200:
                return ""
            return await response.text()
    except Exception:
        return ""


def extract_text_from_html(html: str):
    soup = BeautifulSoup(html, 'html.parser')
    paragraphs = soup.find_all('p')
    text = '\n'.join([p.get_text(strip=True) for p in paragraphs])
    return text[:2000]  # 返回前2000字符用于摘要


# ======================= 主功能函数 =======================
async def get_qwen_answer_async(question: str):
    async with aiohttp.ClientSession() as session:
        print("🔍 正在联网搜索...")
        results = await search_internet(question, session)

        all_texts = []
        for idx, item in enumerate(results[:3], start=1):  # 只处理前3个结果
            print(f"📄 正在处理第 {idx} 个结果：{item['title']}")

            html = await fetch_page_content(item['link'], session)
            if not html:
                continue

            text = extract_text_from_html(html)
            if not text.strip():
                continue

            all_texts.append(text)

        if not all_texts:
            return "⚠️ 未找到与该问题相关的内容。"

        full_context = "\n\n".join(all_texts)[:8000]  # 控制总长度
        qwen = QwenModelConfig()
        answer = await qwen.generate_summary(full_context, question)
        return answer


# ======================= 同步接口 =======================
def get_qwen_answer(question: str):
    """
    同步接口：接收问题，返回 Qwen 搜索+AI 整合后的回答
    """

    # Windows 下使用 selector 策略规避 event loop closed 问题
    if sys.platform == 'win32':
        policy = WindowsSelectorEventLoopPolicy()
        asyncio.set_event_loop_policy(policy)

    loop = asyncio.new_event_loop()
    try:
        answer = loop.run_until_complete(get_qwen_answer_async(question))
        return answer
    finally:
        loop.run_until_complete(asyncio.sleep(0.25))  # 等待资源释放
        loop.close()


# ======================= 示例运行入口 =======================
if __name__ == "__main__":
    user_question = input("请输入你要查询的问题：")
    answer = get_qwen_answer(user_question)

    print("\n=== 最终回答 ===")
    print(answer)