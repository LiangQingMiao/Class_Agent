import os
from openai import OpenAI
from typing import Optional, Tuple

os.environ["DASHSCOPE_API_KEY"] = ""

class SummaryAgent:
    def __init__(self, api_key: Optional[str] = None):
        """
        初始化总结智能体
        
        Args:
            api_key (Optional[str]): 通义千问API密钥，如果不提供则使用环境变量DASHSCOPE_API_KEY
        """
        if api_key is None:
            api_key = os.getenv("DASHSCOPE_API_KEY")
            if not api_key:
                raise ValueError("请设置环境变量 DASHSCOPE_API_KEY 或提供 api_key 参数")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        
    def _create_summary_prompt(self, text: str) -> str:
        """
        创建总结提示词
        
        Args:
            text (str): 要总结的文本
            
        Returns:
            str: 总结提示词
        """
        return f"""请对以下文本进行总结，要求：
1. 保持原文的核心观点和关键信息
2. 使用简洁明了的语言
3. 突出重要的事实和数据
4. 保持逻辑性和连贯性
5. 总结长度控制在原文的1/3左右

文本内容：
{text}"""
    
    def summarize(self, text: str):
        """
        总结文本内容
        
        Args:
            text (str): 要总结的文本
            
        Returns:
            Tuple[bool, str]: (是否成功, 总结结果或错误信息)
        """
        try:
            # 创建提示词
            prompt = self._create_summary_prompt(text)
            
            # 调用API进行总结
            response = self.client.chat.completions.create(
                model="qwen-max",
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个专业的文本总结助手，擅长提取文本的核心内容并进行简洁的总结。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            # 获取总结结果
            summary = response.choices[0].message.content
            
            return summary
            
        except Exception as e:
            return False, f"总结文本时出错: {str(e)}"

def summarize_text(text: str, api_key: Optional[str] = None):
    """
    总结文本的便捷函数
    
    Args:
        text (str): 要总结的文本
        api_key (Optional[str]): 通义千问API密钥，如果不提供则使用环境变量DASHSCOPE_API_KEY
        
    Returns:
        Tuple[bool, str]: (是否成功, 总结结果或错误信息)
    """
    try:
        agent = SummaryAgent(api_key)
        return agent.summarize(text)
    except Exception as e:
        return False

if __name__ == "__main__":
    # 示例用法
    text = """人工智能生成图像检测关键方法综合分析：GFW、NPR 和 UnivFDI
引言：人工智能生成图像检测的必要性
人工智能生成超现实内容的兴起在过去十年中，生成式人工智能（AI）取得了前所未有的进步，以生成对抗网络（GAN）和扩散模型为代表的模型在合成逼真图像方面取得了显著成功。这些生成的图像在很大程度上已达到肉眼难以辨别的程度，对媒体的真实性和信任度构成了严峻挑战。高质量合成媒体的泛滥引发了人们对虚假信息传播、版权侵犯以及在各个领域潜在滥用的严重担忧，这凸显了对强大检测机制的迫切需求。"""
    
    result = summarize_text(text)
    print("总结结果：")
    print(result)
