import streamlit as st

st.set_page_config(
    page_title="ChatGLM3 Demo",
    page_icon=":robot:",
    layout='centered',
    initial_sidebar_state='expanded',
)


import demo_chat
from enum import Enum

DEFAULT_SYSTEM_PROMPT = '''
You are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown.
'''.strip()

# Set the title of the demo
st.title("ChatGLM3 Demo")

# Add your custom text here, with smaller font size
st.markdown(
    "<sub>智谱AI 公开在线技术文档: https://lslfd0slxc.feishu.cn/wiki/WvQbwIJ9tiPAxGk8ywDck6yfnof </sub> \n\n <sub> 更多 ChatGLM3-6B 的使用方法请参考文档。</sub>",
    unsafe_allow_html=True)

# ---------------------------------
from langchain.chains import RetrievalQA
from langchain.prompts.prompt import PromptTemplate
from service.chatglm_service import ChatGLMService
from knowledge_service import KnowledgeService

self.llm_service = ChatGLMService()
self.knowledge_service = KnowledgeService()
#         # 获取大语言模型返回的答案（基于本地知识库查询）
#         def get_knowledeg_based_answer(self, query,
#                                        history_len=5,
#                                        temperature=0.1,
#                                        top_p=0.9,
#                                        top_k=4,
#                                        chat_history=[]):
#             # 定义查询的提示模板格式：
#             prompt_template = '''
#     基于以下已知信息，简洁和专业的来回答用户的问题。
#     如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。
#     已知内容:
#     {context}
#     问题:
#     {question}
#         '''
#             prompt = PromptTemplate(template=prompt_template,
#                                     input_variables=["context", "question"])
#             self.llm_service.history = chat_history[-history_len:] if history_len > 0 else []
#             self.llm_service.temperature = temperature
#             self.llm_service.top_p = top_p
#
#             # 利用预先存在的语言模型、检索器来创建并初始化BaseRetrievalQA类的实例
#             knowledge_chain = RetrievalQA.from_llm(
#                 llm=self.llm_service,
#                 # 基于本地知识库构建一个检索器，并仅返回top_k的结果
#                 retriever=self.knowledge_service.knowledge_base.as_retriever(
#                     search_kwargs={"k": top_k}),
#                 prompt=prompt)
#             # combine_documents_chain的作用是将查询返回的文档内容（page_content）合并到一起作为prompt中context的值
#             # 将combine_documents_chain的合并文档内容改为{page_content}
#
#             knowledge_chain.combine_documents_chain.document_prompt = PromptTemplate(
#                 input_variables=["page_content"], template="{page_content}")
#
#             # 返回结果中是否包含源文档
#             knowledge_chain.return_source_documents = True
#
#             # 传入问题内容进行查询
#             result = knowledge_chain({"query": query})
#             return result
#
#         # 获取大语言模型返回的答案（未基于本地知识库查询）
#         def get_llm_answer(self, query):
#             result = self.llm_service._call(query)
#             return result

# ---------------------------------
class Mode(str, Enum):
    CHAT = '💬 Chat'


with st.sidebar:
    top_p = st.slider(
        'top_p', 0.0, 1.0, 0.8, step=0.01
    )
    temperature = st.slider(
        'temperature', 0.0, 1.5, 0.95, step=0.01
    )
    repetition_penalty = st.slider(
        'repetition_penalty', 0.0, 2.0, 1.1, step=0.01
    )
    max_new_token = st.slider(
        'Output length', 5, 32000, 256, step=1
    )

    cols = st.columns(2)
    export_btn = cols[0]
    clear_history = cols[1].button("Clear History", use_container_width=True)
    retry = export_btn.button("Retry", use_container_width=True)

    system_prompt = st.text_area(
        label="System Prompt (Only for chat mode)",
        height=300,
        value=DEFAULT_SYSTEM_PROMPT,
    )

prompt_text = st.chat_input(
    'Chat with ChatGLM3!',
    key='chat_input',
)

tab = st.radio(
    'Mode',
    [mode.value for mode in Mode],
    horizontal=True,
    label_visibility='hidden',
)

if clear_history or retry:
    prompt_text = ""

match tab:
    case Mode.CHAT:
        demo_chat.main(
            retry=retry,
            top_p=top_p,
            temperature=temperature,
            prompt_text=prompt_text,
            system_prompt=system_prompt,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_token
        )

    case _:
        st.error(f'Unexpected tab: {tab}')
