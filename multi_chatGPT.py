import os  
import json  
import requests    
from azure.search.documents import SearchClient    
from azure.core.credentials import AzureKeyCredential    
from azure.core.pipeline.transport import RequestsTransport    
import streamlit as st    
from openai import AzureOpenAI    
import threading    
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient    
import certifi    
import re  # 正規表現モジュールをインポート  
  
    
# Azure OpenAI設定    
client = AzureOpenAI(    
    api_key=os.getenv("AZURE_OPENAI_KEY"),    
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),    
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")    
)    
    
# Azure Cognitive Search設定    
search_service_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")    
search_service_key = os.getenv("AZURE_SEARCH_KEY")    
index_name = "hatakeyama-l8"    
    
# 'certifi'の証明書バンドルを使用するように設定    
transport = RequestsTransport(verify=certifi.where())    
    
# Azure Searchクライアントの設定    
search_client = SearchClient(    
    endpoint=search_service_endpoint,    
    index_name=index_name,    
    credential=AzureKeyCredential(search_service_key),    
    transport=transport    
)    
    
# Azure Blob Storageクライアントの設定    
blob_service_client = BlobServiceClient.from_connection_string(os.getenv("AZURE_STORAGE_CONNECTION_STRING"))    
container_name = "chat-history-test"    
container_client = blob_service_client.get_container_client(container_name)    
    
st.title("Azure OpenAI ChatGPT with RAG")    
    
# チャット履歴の保存と読み込みの関数    
lock = threading.Lock()    
    
def save_chat_history():    
    with lock:    
        try:    
            blob_client = container_client.get_blob_client("chat_history_test.json")    
            blob_client.upload_blob(json.dumps(st.session_state.sidebar_messages), overwrite=True)    
        except Exception as e:    
            st.error(f"Error saving chat history: {e}")    
    
def load_chat_history():    
    with lock:    
        try:    
            blob_client = container_client.get_blob_client("chat_history_test.json")    
            if blob_client.exists():    
                blob_data = blob_client.download_blob().readall()    
                return json.loads(blob_data)    
        except Exception as e:    
            st.error(f"Error loading chat history: {e}")    
    return []    
    
# セッションの初期化    
if "sidebar_messages" not in st.session_state:    
    st.session_state.sidebar_messages = load_chat_history()    
if "main_chat_messages" not in st.session_state:    
    st.session_state.main_chat_messages = []    
if "current_chat_index" not in st.session_state:    
    st.session_state.current_chat_index = None    
if "show_all_history" not in st.session_state:    
    st.session_state.show_all_history = False    
    
# 新しいチャットを追加する関数    
def start_new_chat():    
    new_chat = {    
        "messages": [],    
        "first_assistant_message": ""    
    }    
    st.session_state.sidebar_messages.append(new_chat)    
    st.session_state.current_chat_index = len(st.session_state.sidebar_messages) - 1    
    
# アプリ起動直後に新しいチャットを自動的に作成    
if st.session_state.current_chat_index is None:    
    start_new_chat()    
    
# アシスタントの最初の回答を要約する関数    
def summarize_text(text, max_length=10):    
    return text[:max_length] + '...' if len(text) > max_length else text    
    
# LaTeX数式が含まれているかをチェックし、適切に表示する関数  
def display_message(content):
    # ブロック数式($$ ... $$)を優先的に処理
    block_latex_matches = re.findall(r"\$\$(.*?)\$\$", content, flags=re.DOTALL)
    if block_latex_matches:
        # ブロック部分でスプリット
        parts = re.split(r"\$\$.*?\$\$", content, flags=re.DOTALL)
        for i, part in enumerate(parts):
            # テキスト部分はそのまま表示（Markdownでインライン数式も処理可能）
            if part.strip():
                st.markdown(part)
            # ブロック数式部分はst.latexで表示（$は不要）
            if i < len(block_latex_matches):
                st.latex(block_latex_matches[i].strip())
    else:
        # ブロック数式がなければ、そのままmarkdownで表示
        st.markdown(content)
    
# サイドバー    
with st.sidebar:    
    st.header("チャット履歴")    
    max_displayed_history = 5    
    max_total_history = 20    
    
    sidebar_messages = [(index, chat) for index, chat in enumerate(st.session_state.sidebar_messages) if chat.get("first_assistant_message")]    
    sidebar_messages = sidebar_messages[::-1]    
    
    if not st.session_state.show_all_history:    
        sidebar_messages = sidebar_messages[:max_displayed_history]    
    else:    
        sidebar_messages = sidebar_messages[:max_total_history]    
    
    for i, (original_index, chat) in enumerate(sidebar_messages):    
        if chat and "first_assistant_message" in chat:    
            keyword = summarize_text(chat["first_assistant_message"])    
            if st.button(keyword, key=f"history_{i}"):    
                st.session_state.current_chat_index = original_index    
                st.session_state.main_chat_messages = []    
                st.session_state.main_chat_messages = st.session_state.sidebar_messages[st.session_state.current_chat_index]["messages"].copy()    
                st.rerun()    
    
    if len(st.session_state.sidebar_messages) > max_displayed_history:    
        if st.session_state.show_all_history:    
            if st.button("少なく表示"):    
                st.session_state.show_all_history = False    
                st.rerun()    
        else:    
            if st.button("もっと見る"):    
                st.session_state.show_all_history = True    
                st.rerun()    
    
    if st.button("新しいチャット"):    
        st.session_state.main_chat_messages = []    
        start_new_chat()    
        st.session_state.show_all_history = False    
        st.rerun()    
    
    past_message_count = st.slider("過去メッセージの数", min_value=1, max_value=20, value=10)    
    
    st.header("検索設定")    
    topNDocuments = st.slider("取得するドキュメント数", min_value=1, max_value=10, value=5)    
    strictness = st.slider("厳密度 (スコアの閾値)", min_value=0.0, max_value=5.0, value=0.1, step=0.1)    
    
# メインエリア    
for message in st.session_state.main_chat_messages:    
    with st.chat_message(message["role"]):    
        display_message(message["content"])    
    
def keyword_semantic_search(query, topNDocuments=5, strictness=0.1):    
    results = search_client.search(    
        search_text=query,    
        select="content",    
        query_type="semantic",    
        semantic_configuration_name="default",    
        query_caption="extractive",    
        query_answer="extractive",    
        top=topNDocuments    
    )    
    
    return [result for result in results if result['@search.score'] >= strictness]    
    
if prompt := st.chat_input("ご質問を入力してください:"):    
    st.session_state.main_chat_messages.append({"role": "user", "content": prompt})    
    with st.chat_message("user"):    
        st.markdown(prompt)    
    
    # キーワードとセマンティック検索を実行    
    search_results = keyword_semantic_search(prompt, topNDocuments=topNDocuments, strictness=strictness)    
    
    # コンテキストの作成    
    context = "\n".join([result['content'] for result in search_results])    
    
    # プロンプト前後にルール文を挿入
    rule_message = ("以下のルールに従って回答してください：\n"
                    "1. LaTeX数式を使用する場合、ブロック数式は $$ ... $$ を使用してください。\n"
                    "2. インライン数式は $ ... $ を使用してください。\n"
                    "3. `\\[ ... \\]` の表記は使用しないでください。\n")
    
    # メッセージリストの作成（RAGコンテキストを含む）    
    num_messages_to_include = past_message_count * 2    
    messages = []
    # ルール文を会話の最初に挿入
    messages.append({"role": "user", "content": rule_message})
    messages.append({"role": "user", "content": f"以下のコンテキストを参考にしてください: {context[:10000]}"})
    messages.extend([{"role": m["role"], "content": m["content"][:500]} for m in st.session_state.main_chat_messages[-(num_messages_to_include):]])
    # ルール文を会話の最後にも挿入
    # messages.append({"role": "user", "content": rule_message})
    
    try:    
        # Azure OpenAIからの応答を取得    
        response = client.chat.completions.create(    
            model="o1-preview",    
            messages=messages    
        )    
    
        # 応答の表示    
        assistant_response = response.choices[0].message.content    
        st.session_state.main_chat_messages.append(    
            {"role": "assistant", "content": assistant_response})    
        with st.chat_message("assistant"):    
            display_message(assistant_response)    
    
        # サイドバーのチャット履歴を更新    
        if st.session_state.current_chat_index is not None:    
            st.session_state.sidebar_messages[st.session_state.current_chat_index]["messages"] = st.session_state.main_chat_messages.copy()    
            if not st.session_state.sidebar_messages[st.session_state.current_chat_index]["first_assistant_message"]:    
                st.session_state.sidebar_messages[st.session_state.current_chat_index]["first_assistant_message"] = assistant_response    
        save_chat_history()    
    except Exception as e:    
        st.error(f"Error: {e}")    
    
# 初期化時にメインチャットメッセージをロード    
if st.session_state.current_chat_index is not None and not st.session_state.main_chat_messages:    
    st.session_state.main_chat_messages = st.session_state.sidebar_messages[st.session_state.current_chat_index]["messages"].copy()    
    
# 新しいチャットがメッセージを持っている場合のみ履歴に保存    
def update_sidebar_messages():    
    if st.session_state.current_chat_index is not None and st.session_state.main_chat_messages:    
        st.session_state.sidebar_messages[st.session_state.current_chat_index]["messages"] = st.session_state.main_chat_messages