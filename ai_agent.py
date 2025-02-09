import os
import re
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="test")

def get_board_description(board, is_red=True):
    """Convert raw board string to text description"""
    piece_map = {
        'R': '车', 'H': '马', 'E': '相', 'A': '仕', 'K': '帅',
        'C': '炮', 'P': '兵', 'r': '車', 'h': '馬', 'e': '象', 
        'a': '士', 'k': '將', 'c': '砲', 'p': '卒', '.': '＋'
    }
    # 获取有效棋盘内容（排除边界行）
    rows = board.split('\n')[2:12]
    
    # 根据玩家视角调整行顺序
    if not is_red:
        rows = rows[::-1]  # 黑方视角需要反转行顺序
        
    desc = []
    for idx, row in enumerate(rows):
        translated = ''.join([piece_map.get(c, c) for c in row[1:]])
        line_num = 9 - idx
        desc_line = f"{line_num} {translated}"
        desc.append(desc_line)
    
    # 根据玩家视角添加坐标标记方向
    coord_line = "  a b c d e f g h i" if is_red else "  i h g f e d c b a"
    desc.append(coord_line)
    return '\n'.join(desc)

def get_ai_move(board_state, is_red=True):
    """Get AI move using LLM API"""
    prompt = f"""你是一个中国象棋大师，请根据当前棋盘状态给出最佳走法。使用坐标格式（如h2e2）回答，只需返回移动坐标，不要其他内容。

当前棋盘状态（你执{"帅" if is_red else "將"}所在一方（棋盘下半部分），不要移动另一方的棋子）：
{get_board_description(board_state, is_red)}

请给出你的走法："""
    # print(prompt)
    try:
        response = client.chat.completions.create(
            model="deepseek-v3",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            stream=True,
        )
        
        # 初始化日志并记录用户prompt
        with open("llm.log", "a", encoding="utf-8") as f:
            f.write(f"=== User Prompt ===\n{prompt}\n\n")
            f.write(f"=== AI Response ===\n")
        
        # 实时记录流式响应到日志
        move_str = ""
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content:
                with open("llm.log", "a", encoding="utf-8") as f:
                    f.write(delta.content)
                move_str += delta.content
        
        # 完整响应后追加换行分隔        
        with open("llm.log", "a", encoding="utf-8") as f:
            f.write("\n\n==========\n\n")
            
        # 移除所有thinking标签
        clean_move = re.sub(r'<thinking>.*?</thinking>', '', move_str, flags=re.DOTALL).strip()
        # 提取移动指令
        if match := re.search(r'([a-i][0-9][a-i][0-9])', clean_move):
            move_str = match.group(1)
        else:
            move_str = clean_move  # 保留原内容用于错误提示
        
        move_str = move_str.strip()
        if re.match(r"^[a-i][0-9][a-i][0-9]$", move_str): 
            return move_str
        raise ValueError("Invalid move format")
    except Exception as e:
        # print(f"AI Error: {e}")
        return None
