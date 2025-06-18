# pip install strip_markdown

import strip_markdown
import textwrap

def strip_md(text:str)->str:
    """去除输入text中的Markdown语法"""
    text = textwrap.dedent(text)
    return strip_markdown.strip_markdown(text).strip()

if __name__ == "__main__":
    print()
    markdown_text = """
    图神经网络（GNN）近年来取得了显著的进展，成为处理图结构数据的重要工具。随着研究界对GNN的关注度迅速上升，其 在学术和工业领域的应用不断扩展。从2017年到2019年，涉及GNN的研究论文年均增长率达到447%，并在ICLR和NeurIPS等顶级人工智能会议中占据重要地位。GNN的核心优势在于能够有效处理非欧几里得结构的数据，捕捉实体间的复杂关系，从而在多个领域实现突破。例如 ，Uber Eats利用GNN改进推荐系统，在关键指标上实现了超过20%的性能提升，AUC指标更是从78%跃升至87%。此外，GNN还在药物发现、 流量预测、Pinterest的内容推荐等领域展现出卓越的能力。这些进展表明，GNN不仅推动了图结构学习的发展，也在更广泛的人工智能领域中扮演着越来越重要的角色。
    """
    plain_text = strip_md(markdown_text)
    print(plain_text)