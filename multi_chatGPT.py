# 必要なライブラリのインポート  
import os  
import json  
import base64  
import threading  
import certifi  
import re  
  
from azure.search.documents import SearchClient  
from azure.core.credentials import AzureKeyCredential  
from azure.core.pipeline.transport import RequestsTransport  
from azure.storage.blob import BlobServiceClient  
  
import streamlit as st  
from openai import AzureOpenAI  
from PIL import Image  
  
# ユーザーIDとユーザー名を取得する関数  
def get_user_identity():  
    try:  
        # Streamlitバージョンによる異なるインポート  
        try:  
            from streamlit.runtime.scriptrunner import get_script_run_ctx  # Streamlit >=1.12.0  
        except ImportError:  
            from streamlit.script_run_context import get_script_run_ctx  # 以前のバージョン  
  
        ctx = get_script_run_ctx()  
        if ctx is None:  
            return None, None  
  
        headers = ctx.request.headers  
        user_header = headers.get('x-ms-client-principal')  
        if not user_header:  
            return None, None  
  
        decoded = base64.b64decode(user_header)  
        user_info = json.loads(decoded.decode('utf-8'))  
        user_id = user_info.get('userId')  
        user_name = user_info.get('userDetails')  
        return user_id, user_name  
    except Exception as e:  
        st.error(f"ユーザー情報の取得中にエラーが発生しました: {e}")  
        return None, None  
  
user_id, user_name = get_user_identity()  
  
if user_id is None:  
    st.error("ユーザー情報を取得できませんでした。Entra ID認証が正しく設定されているか確認してください。")  
    st.stop()  
else:  
    st.write(f"**ログインユーザー:** {user_name}")  
  
# Azure OpenAI設定  
try:  
    client = AzureOpenAI(  
        api_key=os.getenv("AZURE_OPENAI_KEY"),  
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),  
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")  
    )  
except Exception as e:  
    st.error(f"Azure OpenAIクライアントの初期化中にエラーが発生しました: {e}")  
    st.stop()  
  
# Azure Cognitive Search設定  
try:  
    search_service_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")  
    search_service_key = os.getenv("AZURE_SEARCH_KEY")  
    index_name = "hatakeyama-l8"  # ご自身のインデックス名に置き換えてください  
  
    # 'certifi'の証明書バンドルを使用するように設定  
    transport = RequestsTransport(verify=certifi.where())  
  
    # Azure Searchクライアントの設定  
    search_client = SearchClient(  
        endpoint=search_service_endpoint,  
        index_name=index_name,  
        credential=AzureKeyCredential(search_service_key),  
        transport=transport  
    )  
except Exception as e:  
    st.error(f"Azure Cognitive Searchクライアントの初期化中にエラーが発生しました: {e}")  
    st.stop()  
  
# Azure Blob Storageクライアントの設定  
try:  
    blob_service_client = BlobServiceClient.from_connection_string(os.getenv("AZURE_STORAGE_CONNECTION_STRING"))  
    container_name = "chat-history-test"  # ご自身のコンテナ名に置き換えてください  
    container_client = blob_service_client.get_container_client(container_name)  
except Exception as e:  
    st.error(f"Azure Blob Storageクライアントの初期化中にエラーが発生しました: {e}")  
    st.stop()  
  
st.title("Azure OpenAI ChatGPT with Image Upload and RAG")  
  
# チャット履歴のファイル名を固定  
CHAT_HISTORY_FILE_NAME = "chat_history_1.json"  
  
# チャット履歴の保存と読み込みのためのロック  
lock = threading.Lock()  
  
# チャット履歴を保存する関数  
def save_chat_history():  
    with lock:  
        try:  
            blob_client = container_client.get_blob_client(CHAT_HISTORY_FILE_NAME)  
            # 既存の全ユーザーのチャット履歴を読み込む  
            if blob_client.exists():  
                existing_data = blob_client.download_blob().readall()  
                all_chat_histories = json.loads(existing_data)  
            else:  
                all_chat_histories = {}  
  
            # 現在のユーザーの履歴を更新  
            all_chat_histories[user_id] = st.session_state.sidebar_messages  
  
            # 更新した全ユーザーの履歴を保存  
            data = json.dumps(all_chat_histories)  
            blob_client.upload_blob(data, overwrite=True)  
        except Exception as e:  
            st.error(f"チャット履歴の保存中にエラーが発生しました: {e}")  
  
# チャット履歴を読み込む関数  
def load_chat_history():  
    with lock:  
        try:  
            blob_client = container_client.get_blob_client(CHAT_HISTORY_FILE_NAME)  
            if blob_client.exists():  
                blob_data = blob_client.download_blob().readall()  
                all_chat_histories = json.loads(blob_data)  
                # 現在のユーザーの履歴のみを返す  
                return all_chat_histories.get(user_id, [])  
            else:  
                return []  
        except Exception as e:  
            st.error(f"チャット履歴の読み込み中にエラーが発生しました: {e}")  
            return []  
  
# セッションの初期化  
try:  
    if "sidebar_messages" not in st.session_state:  
        st.session_state.sidebar_messages = load_chat_history()  
    if "main_chat_messages" not in st.session_state:  
        st.session_state.main_chat_messages = []  
    if "images" not in st.session_state:  
        st.session_state.images = []  
    if "current_chat_index" not in st.session_state:  
        st.session_state.current_chat_index = None  
    if "show_all_history" not in st.session_state:  
        st.session_state.show_all_history = False  
except Exception as e:  
    st.error(f"セッションの初期化中にエラーが発生しました: {e}")  
    st.stop()  
  
# デフォルトのシステムメッセージを設定  
if 'default_system_message' not in st.session_state:  
    st.session_state['default_system_message'] = "あなたは親切なAIアシスタントです。ユーザーの質問に簡潔かつ正確に答えてください。"  
if "system_message" not in st.session_state:  
    st.session_state.system_message = st.session_state['default_system_message']  
  
# 新しいチャットを開始する関数  
def start_new_chat():  
    try:  
        new_chat = {  
            "role": "system",  
            "content": "New chat started",  
            "messages": [],  
            "first_assistant_message": "",  
            "system_message": st.session_state['default_system_message']  
        }  
        st.session_state.sidebar_messages.append(new_chat)  
        st.session_state.current_chat_index = len(st.session_state.sidebar_messages) - 1  
        st.session_state.system_message = new_chat["system_message"]  
    except Exception as e:  
        st.error(f"新しいチャットの開始中にエラーが発生しました: {e}")  
  
# 初回起動時に新しいチャットを自動的に作成  
if st.session_state.current_chat_index is None:  
    start_new_chat()  
  
# 画像をbase64エンコードする関数  
def encode_image(image_file):  
    try:  
        return base64.b64encode(image_file.getvalue()).decode('utf-8')  
    except Exception as e:  
        st.error(f"画像のエンコード中にエラーが発生しました: {e}")  
        return None  
  
# アシスタントの最初の回答を要約する関数  
def summarize_text(text, max_length=10):  
    if not text:  
        return ''  
    return text[:max_length] + '...' if len(text) > max_length else text  
  
# サイドバー  
with st.sidebar:  
    st.header("システムメッセージ設定")  
    try:  
        if st.session_state.current_chat_index is not None:  
            current_system_message = st.session_state.sidebar_messages[st.session_state.current_chat_index].get("system_message", st.session_state['default_system_message'])  
        else:  
            current_system_message = st.session_state['default_system_message']  
  
        system_message_key = f"system_message_{st.session_state.current_chat_index}"  
        system_message = st.text_area(  
            "システムメッセージを入力してください",  
            value=current_system_message,  
            height=100,  
            key=system_message_key  
        )  
  
        if st.session_state.current_chat_index is not None:  
            st.session_state.sidebar_messages[st.session_state.current_chat_index]["system_message"] = system_message  
        st.session_state.system_message = system_message  
  
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
                    st.session_state.images = []  
                    st.session_state.main_chat_messages = st.session_state.sidebar_messages[st.session_state.current_chat_index]["messages"].copy()  
                    st.session_state.system_message = st.session_state.sidebar_messages[st.session_state.current_chat_index].get("system_message", st.session_state['default_system_message'])  
                    system_message_key = f"system_message_{st.session_state.current_chat_index}"  
                    if system_message_key in st.session_state:  
                        del st.session_state[system_message_key]  
                    st.experimental_rerun()  
  
        if len(st.session_state.sidebar_messages) > max_displayed_history:  
            if st.session_state.show_all_history:  
                if st.button("少なく表示"):  
                    st.session_state.show_all_history = False  
                    st.experimental_rerun()  
            else:  
                if st.button("もっと見る"):  
                    st.session_state.show_all_history = True  
                    st.experimental_rerun()  
  
        if st.button("新しいチャット"):  
            st.session_state.main_chat_messages = []  
            st.session_state.images = []  
            start_new_chat()  
            st.session_state.show_all_history = False  
            st.experimental_rerun()  
  
        st.header("画像アップロード")  
        uploaded_files = st.file_uploader("画像を選択してください", type=["jpg", "jpeg", "png"], key="uploader", accept_multiple_files=True)  
        if uploaded_files:  
            for uploaded_file in uploaded_files:  
                try:  
                    image = Image.open(uploaded_file)  
                    encoded_image = encode_image(uploaded_file)  
                    if not any(img["name"] == uploaded_file.name for img in st.session_state.images):  
                        st.session_state.images.append({  
                            "image": image,  
                            "encoded": encoded_image,  
                            "name": uploaded_file.name  
                        })  
                    st.success(f"画像 '{uploaded_file.name}' がアップロードされました。")  
                except Exception as e:  
                    st.error(f"画像 '{uploaded_file.name}' の処理中にエラーが発生しました: {e}")  
  
        st.subheader("アップロードされた画像")  
        for idx, img_data in enumerate(st.session_state.images):  
            st.image(img_data["image"], caption=img_data["name"], use_column_width=True)  
            if st.button(f"削除 {img_data['name']}", key=f"delete_{idx}"):  
                st.session_state.images.pop(idx)  
                st.experimental_rerun()  
  
        past_message_count = st.slider("過去メッセージの数", min_value=1, max_value=20, value=10)  
        temperature = st.slider("温度", min_value=0.0, max_value=1.0, value=0.5, step=0.1)  
  
        st.header("検索設定")  
        topNDocuments = st.slider("取得するドキュメント数", min_value=1, max_value=10, value=5)  
        strictness = st.slider("厳密度 (スコアの閾値)", min_value=0.0, max_value=5.0, value=0.1, step=0.1)  
  
    except Exception as e:  
        st.error(f"サイドバーの処理中にエラーが発生しました: {e}")  
  
# メインエリア  
try:  
    for message in st.session_state.main_chat_messages:  
        with st.chat_message(message["role"]):  
            st.markdown(message["content"])  
except Exception as e:  
    st.error(f"チャットメッセージの表示中にエラーが発生しました: {e}")  
  
# セマンティック検索を行う関数  
def keyword_semantic_search(query, topNDocuments=5, strictness=0.1):  
    try:  
        results = search_client.search(  
            search_text=query,  
            select="content",  
            query_type="semantic",  
            semantic_configuration_name="default",  
            query_caption="extractive",  
            query_answer="extractive",  
            top=topNDocuments  
        )  
        return [result for result in results if result.get('@search.score', 0) >= strictness]  
    except Exception as e:  
        st.error(f"検索中にエラーが発生しました: {e}")  
        return []  
  
# ユーザーからの入力を処理  
prompt = st.chat_input("ご質問を入力してください:")  
if prompt:  
    try:  
        st.session_state.main_chat_messages.append({"role": "user", "content": prompt})  
        with st.chat_message("user"):  
            st.markdown(prompt)  
  
        # キーワードとセマンティック検索を実行  
        search_results = keyword_semantic_search(prompt, topNDocuments=topNDocuments, strictness=strictness)  
  
        # コンテキストの作成  
        context = "\n".join([result['content'] for result in search_results])  
  
        # メッセージリストの作成（コンテキストを含む）  
        num_messages_to_include = past_message_count * 2  
        messages = [{"role": "system", "content": st.session_state.system_message}]  
        messages.extend([{"role": m["role"], "content": m["content"][:500]} for m in st.session_state.main_chat_messages[-(num_messages_to_include):]])  
        messages.insert(1, {"role": "system", "content": f"以下のコンテキストを使用してユーザーを支援してください: {context[:1000]}"})  
  
        # Azure OpenAIからの応答を取得  
        response = client.chat.completions.create(  
            model="pm-GPT4o",  
            messages=messages,  
            temperature=temperature  
        )  
  
        # 応答の表示  
        assistant_response = response.choices[0].message.content  
        st.session_state.main_chat_messages.append({"role": "assistant", "content": assistant_response})  
  
        with st.chat_message("assistant"):  
            st.markdown(assistant_response)  
  
        # サイドバーのチャット履歴を更新  
        if st.session_state.current_chat_index is not None:  
            st.session_state.sidebar_messages[st.session_state.current_chat_index]["messages"] = st.session_state.main_chat_messages.copy()  
            if not st.session_state.sidebar_messages[st.session_state.current_chat_index]["first_assistant_message"]:  
                st.session_state.sidebar_messages[st.session_state.current_chat_index]["first_assistant_message"] = assistant_response  
  
        save_chat_history()  
  
    except Exception as e:  
        st.error(f"処理中にエラーが発生しました: {e}")  
  
# メインチャットメッセージがまだロードされていない場合にロード  
if st.session_state.current_chat_index is not None and not st.session_state.main_chat_messages:  
    try:  
        st.session_state.main_chat_messages = st.session_state.sidebar_messages[st.session_state.current_chat_index]["messages"].copy()  
    except Exception as e:  
        st.error(f"チャットメッセージの読み込み中にエラーが発生しました: {e}")  
  
# チャット履歴を更新する関数  
def update_sidebar_messages():  
    try:  
        if st.session_state.current_chat_index is not None and st.session_state.main_chat_messages:  
            st.session_state.sidebar_messages[st.session_state.current_chat_index]["messages"] = st.session_state.main_chat_messages.copy()  
    except Exception as e:  
        st.error(f"サイドバーメッセージの更新中にエラーが発生しました: {e}")  
  
update_sidebar_messages()  