import streamlit as st
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# ==================== API密钥安全检查 ====================
if 'OPENAI_API_KEY' in st.secrets:
    api_key = st.secrets['OPENAI_API_KEY']
    st.success("✅ API密钥加载成功")
else:
    st.error("❌ 请在Streamlit Cloud的Secrets中配置API密钥")
    st.stop()  # 没有密钥就停止执行
# =======================================================

# 设置网页标题和图标
st.set_page_config(page_title="英文表达参考助手", page_icon="🌍")
st.title("🌍 英文表达参考助手")
st.markdown("输入中文，获取多种情景的英文表达参考")

# 在侧边栏中添加风格选择
with st.sidebar:
    st.header("🎛️ 设置")
    
    # 使用session_state保存选择
    if "style_preference" not in st.session_state:
        st.session_state.style_preference = "综合推荐"
    
    style_preference = st.selectbox(
        "偏好风格",
        options=["综合推荐", "口语交流", "商务书面", "学术写作", "情感表达"],
        index=["综合推荐", "口语交流", "商务书面", "学术写作", "情感表达"].index(st.session_state.style_preference)
    )
    
    if style_preference != st.session_state.style_preference:
        st.session_state.style_preference = style_preference
        st.success(f"已切换到: {style_preference}风格")

# 聊天输入框
user_input = st.chat_input("输入您想表达的中文内容...")

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "您好！我是英文表达参考助手，请输入中文内容，我会为您提供多种情景的英文表达参考。"}
    ]

# 显示对话历史
for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

# 获取提示词模板
def get_expression_prompt(style_preference):
    """根据风格偏好获取提示词模板"""
    
    style_instructions = {
        "综合推荐": "提供3-4种不同风格的英文表达，包括口语、书面和情感表达版本",
        "口语交流": "重点提供自然、地道的口语表达，适合日常对话使用",
        "商务书面": "侧重正式、专业的商务和书面表达，注意用词准确",
        "学术写作": "提供学术论文、正式文档中使用的严谨表达",
        "情感表达": "强调情感色彩和语气，提供不同情感强度的表达方式"
    }
    
    system_template = """你是专业的英文表达顾问，专门帮助中文用户找到最适合情景的英文表达。

## 你的任务：
用户输入中文句子，你需要提供多种英文表达方式，每种表达都要：
1. 标注适用场景和风格特点
2. 提供中文回译说明细微差别
3. 给出语法要点和用词分析
4. 最后给出综合推荐

## 输出格式要求：
⸻
[中文原句]
这句话的英文可以这样表达👇

✅ [风格1名称]
[英文表达1]
（[中文回译说明细微差别]）

⸻

✅ [风格2名称] 
[英文表达2]
（[中文回译说明细微差别]）

⸻

✅ [风格3名称]
[英文表达3]
（[中文回译说明细微差别]）

⸻

💡 语法要点：
• [要点1]
• [要点2]

⸻

🪄 总结推荐：
✅ [最推荐的表达]
[推荐理由]

请根据用户偏好侧重：{style_instruction}"""

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", "{input}"),
    ])
    
    return prompt_template, style_instructions[style_preference]

# 生成英文表达参考
def generate_expression_reference(user_input, style_preference):
    """生成多种英文表达参考"""
    
    # 获取提示词模板和风格说明
    prompt_template, style_instruction = get_expression_prompt(style_preference)
    
    # 创建模型客户端 - 使用安全的api_key变量
    client = ChatOpenAI(
        api_key=api_key,  # ← 改为使用上面定义的api_key变量
        model="deepseek-chat",
        base_url="https://api.deepseek.com",
        temperature=0.3,  # 稍高的温度以获得更多创造性表达
    )
    
    # 创建对话链
    chain = LLMChain(llm=client, prompt=prompt_template)
    
    # 生成回复
    response = chain.run(
        input=user_input,
        style_instruction=style_instruction
    )
    
    return response

# 处理用户输入
if user_input:
    # 显示用户消息
    st.chat_message("human").write(user_input)
    st.session_state.messages.append({"role": "human", "content": user_input})
    
    # 生成英文表达参考
    with st.spinner("正在生成多种英文表达参考..."):
        try:
            response = generate_expression_reference(user_input, st.session_state.style_preference)
            
            # 显示AI回复
            st.chat_message("assistant").write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # 添加复制功能提示
            st.sidebar.success("💡 提示：可以复制您喜欢的表达方式")
            
        except Exception as e:
            error_msg = f"生成表达时出现错误，请重试: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

# 在侧边栏添加使用指南
with st.sidebar:
    st.divider()
    st.subheader("📚 使用指南")
    st.markdown("""
    **如何使用：**
    1. 在输入框输入中文句子
    2. 选择偏好的表达风格
    3. 查看多种英文表达参考
    4. 选择最适合您情景的表达
    
    **适用场景：**
    • 日常口语交流
    • 商务邮件写作  
    • 学术论文表达
    • 情感表达优化
    """)
    
    # 显示示例
    st.divider()
    st.subheader("🎯 示例输入")
    st.code("""
我以为他们会感到沮丧，因为下雨，不能外出。
今天的工作进展很顺利。
这个想法听起来很有创意。
    """, language="text")