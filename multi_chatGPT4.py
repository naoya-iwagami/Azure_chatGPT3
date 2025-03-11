import os  
import json  
import base64  
import threading  
import datetime  
import uuid  
import streamlit as st  
from azure.search.documents import SearchClient  
from azure.core.credentials import AzureKeyCredential  
from azure.core.pipeline.transport import RequestsTransport  
from azure.cosmos import CosmosClient, PartitionKey  
from openai import AzureOpenAI  
from PIL import Image  
import certifi  
import tiktoken  

# Azure OpenAIの設定  
client = AzureOpenAI(  
    api_key=os.getenv("AZURE_OPENAI_KEY"),  
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),  
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")  
)  
  
# Azure Cognitive Searchの設定  
search_service_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")  
search_service_key = os.getenv("AZURE_SEARCH_KEY")  
  
# 'certifi'の証明書バンドルを使用するように設定  
transport = RequestsTransport(verify=certifi.where())  
  
# Azure Cosmos DBの設定  
cosmos_endpoint = os.getenv("AZURE_COSMOS_ENDPOINT")  
cosmos_key = os.getenv("AZURE_COSMOS_KEY")  
database_name = 'chatdb'  
container_name = 'chathistory'  
  
cosmos_client = CosmosClient(cosmos_endpoint, credential=cosmos_key)  
database = cosmos_client.get_database_client(database_name)  
container = database.get_container_client(container_name)  
  
# ロックオブジェクトの初期化  
lock = threading.Lock()  
  
def main():  
    # ユーザー情報の取得  
    if hasattr(st, 'experimental_user'):  
        user = st.experimental_user  
        if user and 'email' in user:  
            user_id = user['email']  # ユーザーのメールアドレスを使用  
            st.session_state["user_id"] = user_id  
        else:  
            st.error("ユーザー情報を取得できませんでした。")  
            st.stop()  
    else:  
        st.error("ユーザー情報を取得できませんでした。")  
        st.stop()  
  
    # タイトル  
    st.title("Azure OpenAI ChatGPT with Image Upload and RAG")  
  
    # サイドバー: モデル選択  
    st.sidebar.header("モデル選択")  
    model_to_use = st.sidebar.selectbox(  
        "使用するモデルを選択してください",  
        options=["gpt-4o", "o1"],  
        index=0  # デフォルトで gpt-4o を選択  
    )  
  
    # インデックス選択用のプルダウンを追加  
    st.sidebar.header("インデックス選択")  
    index_options = {  
        "通常データ": "filetest11",  
        "SANUQIメール": "filetest13",  
        "予備１": "filetest14",  
        "予備２": "filetest15"  
    }  
    selected_index_label = st.sidebar.selectbox(  
        "使用するデータを選択してください",  
        list(index_options.keys()),  
        index=0  
    )  
    # 選択されたラベルから実際のインデックス名を取得  
    index_name = index_options[selected_index_label]  
  
    # SearchClient を生成 (選択した index_name を使用)  
    search_client = SearchClient(  
        endpoint=search_service_endpoint,  
        index_name=index_name,  
        credential=AzureKeyCredential(search_service_key),  
        transport=transport  
    )  
  
    # Cosmos DB 保存関数  
    def save_chat_history():  
        with lock:  
            try:  
                user_id = st.session_state["user_id"]  
                item = {  
                    'id': user_id,  # ユーザーIDをドキュメントのIDとして使用  
                    'user_id': user_id,  
                    'sidebar_messages': st.session_state.sidebar_messages,  
                    'timestamp': datetime.datetime.utcnow().isoformat()  
                }  
                container.upsert_item(item)  
            except Exception as e:  
                st.error(f"チャット履歴の保存中にエラーが発生しました: {e}")  
  
    # Cosmos DB 読み込み関数  
    def load_chat_history():  
        with lock:  
            try:  
                user_id = st.session_state["user_id"]  
                item = container.read_item(item=user_id, partition_key=user_id)  
                if item:  
                    return item.get('sidebar_messages', [])  
            except Exception as e:  
                if 'NotFound' in str(e):  
                    return []  # データが存在しない場合は空のリストを返す  
                st.error(f"チャット履歴の読み込み中にエラーが発生しました: {e}")  
            return []  
  
    # セッション状態の初期化  
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
  
    # デフォルトのシステムメッセージ設定  
    if 'default_system_message' not in st.session_state:  
        st.session_state['default_system_message'] = "あなたは親切なAIアシスタントです。ユーザーの質問に簡潔かつ正確に答えてください。"  
  
    if "system_message" not in st.session_state:  
        st.session_state.system_message = st.session_state['default_system_message']  
  
    # 新しいチャットを追加する関数  
    def start_new_chat():  
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
  
    # アプリ起動直後に新しいチャットを自動作成  
    if st.session_state.current_chat_index is None:  
        start_new_chat()  
  
    # 画像をbase64エンコードする関数  
    def encode_image(image_file):  
        return base64.b64encode(image_file.getvalue()).decode('utf-8')  
  
    # アシスタントの最初の回答を要約する関数  
    def summarize_text(text, max_length=10):  
        return text[:max_length] + '...' if len(text) > max_length else text  
  
    # サイドバーの構築 (システムメッセージ、チャット履歴、画像アップロード、検索設定など)  
    with st.sidebar:  
        st.header("システムメッセージ設定")  
        if st.session_state.current_chat_index is not None:  
            current_system_message = st.session_state.sidebar_messages[st.session_state.current_chat_index].get(  
                "system_message", st.session_state['default_system_message'])  
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
  
        # チャット履歴  
        st.header("チャット履歴")  
        max_displayed_history = 5  
        max_total_history = 20  # もっと見るを押したときに表示する最大履歴数  
  
        sidebar_messages = [  
            (index, chat)  
            for index, chat in enumerate(st.session_state.sidebar_messages)  
            if chat.get("first_assistant_message")  
        ]  
        sidebar_messages = sidebar_messages[::-1]  # 新しいものから表示  
  
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
                    st.session_state.main_chat_messages = st.session_state.sidebar_messages[  
                        st.session_state.current_chat_index]["messages"].copy()  
                    st.session_state.system_message = st.session_state.sidebar_messages[  
                        st.session_state.current_chat_index].get("system_message", st.session_state['default_system_message'])  
                    # テキストエリアの値を再同期  
                    system_message_key = f"system_message_{st.session_state.current_chat_index}"  
                    if system_message_key in st.session_state:  
                        del st.session_state[system_message_key]  
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
            st.session_state.images = []  
            start_new_chat()  
            st.session_state.show_all_history = False  
            st.rerun()  
  
        # 画像アップロード  
        st.header("画像アップロード")  
        uploaded_files = st.file_uploader(  
            "画像を選択してください", type=["jpg", "jpeg", "png"],  
            key="uploader", accept_multiple_files=True  
        )  
        if uploaded_files:  
            for uploaded_file in uploaded_files:  
                image = Image.open(uploaded_file)  
                encoded_image = encode_image(uploaded_file)  
                if not any(img["name"] == uploaded_file.name for img in st.session_state.images):  
                    st.session_state.images.append({  
                        "image": image,  
                        "encoded": encoded_image,  
                        "name": uploaded_file.name  
                    })  
                    st.success(f"画像 '{uploaded_file.name}' がアップロードされました。")  
  
        st.subheader("アップロードされた画像")  
        for idx, img_data in enumerate(st.session_state.images):  
            st.image(img_data["image"], caption=img_data["name"], use_container_width=True)  
            if st.button(f"削除 {img_data['name']}", key=f"delete_{idx}"):  
                st.session_state.images.pop(idx)  
                st.rerun()  
  
        # 過去メッセージの数  
        past_message_count = st.slider("過去メッセージの数", min_value=1, max_value=50, value=10)  
  
        # 検索設定  
        st.header("検索設定")  
        topNDocuments = st.slider("取得するドキュメント数", min_value=1, max_value=10, value=5)  
        strictness = st.slider("厳密度 (スコアの閾値)", min_value=0.0, max_value=10.0, value=0.1, step=0.1)  
  
    # 選択されたインデックスに基づく検索用関数をここで定義  
    def keyword_semantic_search(query, topNDocuments=5, strictness=0.1):  
        results = search_client.search(  
            search_text=query,  
            search_fields=["content"],  
            select="content",  
            query_type="semantic",  
            semantic_configuration_name="default",  
            query_caption="extractive",  
            query_answer="extractive",  
            top=topNDocuments  
        )  
        # 結果をしきい値でフィルタ  
        results_list = [result for result in results if result['@search.score'] >= strictness]  
        # スコアの高い順にソート  
        results_list.sort(key=lambda x: x['@search.score'], reverse=True)  
        return results_list  
  
    # トークン数を計算する関数  
    def num_tokens_from_messages(messages, model="model_to_use"):  
        encoding = tiktoken.encoding_for_model(model)  
        num_tokens = 0  
        for message in messages:  
            # role + contentなどのキーごとに4トークン前後になるというざっくり計算  
            num_tokens += 4  
            for key, value in message.items():  
                if key == "content" and isinstance(value, list):  
                    combined_text = ""  
                    for item in value:  
                        if isinstance(item, dict) and "text" in item:  
                            combined_text += item["text"]  
                        else:  
                            combined_text += str(item)  
                    token_str = combined_text  
                else:  
                    token_str = value  
                num_tokens += len(encoding.encode(token_str))  
                if key == "name":  
                    num_tokens -= 1  
            num_tokens += 2  
        return num_tokens  
  
    # メインエリア: チャット履歴の表示  
    for message in st.session_state.main_chat_messages:  
        with st.chat_message(message["role"]):  
            st.markdown(message["content"])  
  
    # ユーザーからの入力  
    if prompt := st.chat_input("ご質問を入力してください:"):  
        st.session_state.main_chat_messages.append({"role": "user", "content": prompt})  
        with st.chat_message("user"):  
            st.markdown(prompt)  
  
        # 関連するデータを検索  
        search_results = keyword_semantic_search(prompt, topNDocuments=topNDocuments, strictness=strictness)  
  
        # コンテキスト作成  
        context = "\n".join([result['content'] for result in search_results])  
  
        # ルールメッセージ  
        rule_message = (  
            "回答する際は、以下のルールに従ってください：\n"  
            "1. 簡潔かつ正確に回答してください。\n"  
            "2. 必要に応じて、提供されたコンテキストを参照してください。\n"  
        )  
  
        # RAGコンテキスト含むメッセージリスト生成  
        num_messages_to_include = past_message_count * 2  
        messages = []  
        messages.append({"role": "system", "content": st.session_state.system_message})  
        messages.append({"role": "user", "content": rule_message})  
        messages.append({"role": "user", "content": f"以下のコンテキストを参考にしてください: {context[:10000]}"})  
        messages.extend([{"role": m["role"], "content": m["content"]} for m in st.session_state.main_chat_messages[-(num_messages_to_include):]])  
  
        # 画像データがあれば追加  
        if st.session_state.images:  
            image_contents = [  
                {  
                    "type": "image_url",  
                    "image_url": {"url": f"data:image/jpeg;base64,{img['encoded']}"}  
                }  
                for img in st.session_state.images  
            ]  
            # RAGコンテキストを入れたメッセージ(messages[2])に追加  
            messages[2]["content"] = [{"type": "text", "text": messages[2]["content"]}] + image_contents  
  
        try:  
            # モデルの最大トークン数（仮に大きく設定しています）  
            max_tokens = 200000  
  
            # 入力トークン数を計算  
            input_token_count = num_tokens_from_messages(messages, model=model_to_use)  
            tokens_remaining = max_tokens - input_token_count  
  
            # Azure OpenAIからの応答  
            response = client.chat.completions.create(  
                model=model_to_use,  
                messages=messages,  
            )  
            assistant_response = response.choices[0].message.content  
  
            # 応答を表示  
            st.session_state.main_chat_messages.append({"role": "assistant", "content": assistant_response})  
            with st.chat_message("assistant"):  
                st.markdown(assistant_response)  
  
            # 検索結果の引用を表示  
            st.write("### 引用文：")  
            if search_results:  
                for i, result in enumerate(search_results):  
                    score = result['@search.score']  
                    with st.expander(f"ドキュメント {i+1}（スコア: {score:.2f}）", expanded=False):  
                        st.caption(result['content'])  
            else:  
                st.write("引用なし")  
  
            # サイドバーのチャット履歴を更新  
            if st.session_state.current_chat_index is not None:  
                st.session_state.sidebar_messages[st.session_state.current_chat_index]["messages"] = st.session_state.main_chat_messages.copy()  
                if not st.session_state.sidebar_messages[st.session_state.current_chat_index]["first_assistant_message"]:  
                    st.session_state.sidebar_messages[st.session_state.current_chat_index]["first_assistant_message"] = assistant_response  
  
            # トークン使用状況の表示  
            if hasattr(response, 'usage'):  
                prompt_tokens = response.usage.prompt_tokens  
                completion_tokens = response.usage.completion_tokens  
                total_tokens = response.usage.total_tokens  
                tokens_remaining = max_tokens - total_tokens  
  
                st.write("### トークン使用状況：")  
                st.write(f"- 入力トークン数: {prompt_tokens}")  
                st.write(f"- 出力トークン数: {completion_tokens}")  
                st.write(f"- 合計トークン数: {total_tokens}")  
                st.write(f"- 残りトークン数: {tokens_remaining}")  
            else:  
                output_token_count = len(tiktoken.encoding_for_model(model_to_use).encode(assistant_response))  
                total_tokens = input_token_count + output_token_count  
                tokens_remaining = max_tokens - total_tokens  
  
                st.write("### トークン使用状況：")  
                st.write(f"- 入力トークン数: {input_token_count}")  
                st.write(f"- 出力トークン数: {output_token_count}")  
                st.write(f"- 合計トークン数: {total_tokens}")  
                st.write(f"- 残りトークン数: {tokens_remaining}")  
  
            # チャット履歴の保存  
            save_chat_history()  
  
        except Exception as e:  
            st.error(f"エラーが発生しました: {e}")  
  
    # アプリ起動後初期ロード  
    if (st.session_state.current_chat_index is not None) and (not st.session_state.main_chat_messages):  
        st.session_state.main_chat_messages = st.session_state.sidebar_messages[  
            st.session_state.current_chat_index  
        ]["messages"].copy()  
  
    def update_sidebar_messages():  
        if st.session_state.current_chat_index is not None and st.session_state.main_chat_messages:  
            st.session_state.sidebar_messages[st.session_state.current_chat_index]["messages"] = st.session_state.main_chat_messages.copy()  
            st.session_state.sidebar_messages[st.session_state.current_chat_index]["system_message"] = st.session_state.system_message  
            save_chat_history()  
  
    # チャット履歴の保存を呼び出す  
    update_sidebar_messages()  
  
if __name__ == "__main__":  
    main()  