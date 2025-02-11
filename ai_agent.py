import os
import re
from openai import OpenAI

# 根据玩家颜色创建不同的客户端配置
def create_llm_client(api_url, api_key, model):
    """Create LLM client with specified configuration"""
    return OpenAI(
        base_url=api_url or "http://127.0.0.1:8080/v1",
        api_key=api_key or "test"
    )

class AIAgent:
    def __init__(self, api_url, api_key, model):
        self.client = create_llm_client(api_url, api_key, model)
        self.model = model or "deepseek-v3"

def get_board_description(board, is_red=True):
    """Convert raw board string to text description"""
    piece_map = {
        'R': '車', 'H': '馬', 'E': '相', 'A': '仕', 'K': '帅',
        'C': '炮', 'P': '兵', 'r': '车', 'h': '马', 'e': '象', 
        'a': '士', 'k': '将', 'c': '砲', 'p': '卒', '.': '〇'
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
    
    # 添加楚河汉界
    desc.insert(5, "    楚河    汉界    ")

    # 根据玩家视角添加坐标标记方向
    coord_line = "  a b c d e f g h i" if is_red else "  i h g f e d c b a"
    desc.append(coord_line)
    return '\n'.join(desc)

def get_ai_move(agent, board_state, last_action, trash_talk, is_red=True, is_initial=False, **kwargs):
    """Get AI move using LLM API"""
    # 获取当前玩家的参数
    key_prefix = 'red' if is_red else 'black'

    prompt = f"""你是一个中国象棋大师，请根据当前棋盘状态分析最佳走法。按照以下格式返回：
<thinking>
1. 给出我方棋子的所有坐标: {"車馬相仕帅仕相馬車炮炮兵兵兵兵兵" if is_red else "车马象士将士象马车砲砲卒卒卒卒卒"}
2. 给出对方棋子的所有坐标: {"车马象士将士象马车砲砲卒卒卒卒卒" if is_red else "車馬相仕帅仕相馬車炮炮兵兵兵兵兵"}
3. 分析当前的布局状态，同时结合短期问题和长期布局考虑，说明走哪一步是合理的
</thinking>
<move>走法坐标（例如h2e2）</move>
# 主动跟对方就当期局势发起简洁且尽量幽默的对话，或者回复对方的聊天，你也可以根据局势尝试劝降对方
<talk>聊天</talk>

我方执{"帅" if is_red else "将"}，在棋盘下半部分，请不要移动对方的棋子。

刚刚对方走了一招：{last_action}，并跟你聊天：{trash_talk}

当前棋盘状态({"刚开局，你先走第一步" if is_initial else "棋局已经进行一段时间了"}):

{get_board_description(board_state, is_red)}

注意，为了区分棋子，{"对方" if is_red else "我方"}的炮用砲表示

棋盘布局一共是十行九列, 棋盘上已经标注了坐标, 行坐标从下到上是0-9, 纵坐标从左到右是{"a-i" if is_red else "i-a"},其中〇表示空位

请分析后给出走法："""
    # print(prompt)
    try:
        response = agent.client.chat.completions.create(
            model=agent.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,  # 提高温度参数增加创造性
            stream=True,
        )
        
        # 初始化日志并记录用户prompt
        with open("llm.log", "a", encoding="utf-8") as f:
            f.write(f"=== Model Info ===\n{agent.model}\n\n")
            f.write(f"=== User Prompt ===\n{prompt}\n\n")
            f.write(f"=== AI Response ===\n")
        
        # 实时记录流式响应到日志
        move_str = ""
        for chunk in response:
            if len(chunk.choices) == 0:
                continue
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
        # 提取各部分内容
        move_match = re.search(r'<move>([a-i][0-9][a-i][0-9])</move>', clean_move)
        trash_talk_match = re.search(r'<talk>(.+?)</talk>', clean_move)

        if move_match:
            move_str = move_match.group(1)
            trash_talk = trash_talk_match.group(1) if trash_talk_match else ""
            return move_str, trash_talk
        
        # 如果新格式解析失败尝试旧格式
        if match := re.search(r'([a-i][0-9][a-i][0-9])', clean_move):
            return match.group(1), ""
            
        raise ValueError(f"Invalid response format: {clean_move}")
    except Exception as e:
        # print(f"AI Error: {e}")
        return None, ""
