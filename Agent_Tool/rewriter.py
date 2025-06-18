from openai import OpenAI
import os
from typing import List, Tuple,Optional
os.environ["DASHSCOPE_API_KEY"] = ""


class QuestionRewriter:
    def __init__(self, api_key: Optional[str] = None):
        """
        初始化问题撰写回答智能体
        
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
        
    def _create_rewrite_prompt(self, questions: List[str]) -> str:
        """
        创建撰写回答提示词
        
        Args:
            questions (List[str]): 原始问题列表
            
        Returns:
            str: 撰写回答提示词
        """
        questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
        return f"""以老师书面口吻回答以下问题。要求：
            1. 使用专业、严谨的学术语言
            2. 保持问题的核心内容不变
            3. 适当使用学术用语和表达方式
            4. 每个问题都应该清晰、完整
            5. 保持问题回答的逻辑性和连贯性

            原始问题列表：
            {questions_text}

            请按照相同的编号格式输出回答后的问题。"""
    
    def rewrite_questions(self, questions: List[str]) -> Tuple[bool, List[str]]:
        """
        回答问题列表
        
        Args:
            questions (List[str]): 原始问题列表
            
        Returns:
            Tuple[bool, List[str]]: (是否成功)
        """
        try:
            # 创建提示词
            prompt = self._create_rewrite_prompt(questions)
            
            # 调用API进行撰写回答
            response = self.client.chat.completions.create(
                model="qwen-max",
                messages=[
                    {
                        "role": "system",
                        "content": "你是一位经验丰富的教师，擅长用专业、严谨的学术语言回答问题。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            # 获取撰写回答结果
            rewritten_text = response.choices[0].message.content
            
            # 解析撰写回答后的问题列表
            rewritten_questions = []
            for line in rewritten_text.split('\n'):
                if line.strip() and line[0].isdigit():
                    # 提取问题内容（去掉编号和点）
                    question = line.split('.', 1)[1].strip()
                    rewritten_questions.append(question)
            
            return True, rewritten_questions
            
        except Exception as e:
            return False, [f"撰写回答问题时出错: {str(e)}"]

def rewrite_questions(questions: List[str], api_key: Optional[str] = None) -> Tuple[bool, List[str]]:
    """
    撰写回答问题列表的便捷函数
    
    Args:
        questions (List[str]): 原始问题列表
        api_key (Optional[str]): 通义千问API密钥，如果不提供则使用环境变量DASHSCOPE_API_KEY
        
    Returns:
        Tuple[bool, List[str]]: (是否成功, 撰写回答后的问题列表或错误信息)
    """
    try:
        agent = QuestionRewriter(api_key)
        return agent.rewrite_questions(questions)
    except Exception as e:
        return False, [f"创建撰写回答智能体时出错: {str(e)}"]

if __name__ == "__main__":
    # 示例用法
    questions = [
        "什么是人工智能？",
        "机器学习的主要类型有哪些？",
        "深度学习与传统机器学习的区别是什么？",
        "神经网络的基本结构是什么？",
        "什么是过拟合？如何避免？"
    ]
    
    success, result = rewrite_questions(questions)
    if success:
        print("撰写回答：")
        for i, question in enumerate(result, 1):
            print(f"{i}. {question}")
    else:
        print(f"撰写回答失败：{result[0]}")