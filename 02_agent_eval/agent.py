from typing import Any, Generator, Optional, Sequence, Union

import mlflow
from databricks_langchain import (
    ChatDatabricks,
    VectorSearchRetrieverTool,
    DatabricksFunctionClient,
    UCFunctionToolkit,
    set_uc_function_client,
)
from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt.tool_node import ToolNode
from mlflow.langchain.chat_agent_langgraph import ChatAgentState
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)

import os
from typing import List, Dict, Any
import json

############################################
# 環境変数から設定を取得
############################################
def get_env_config():
    """環境変数から必要最小限の設定を取得"""
    
    # 必須の環境変数を取得
    llm_endpoint = os.environ.get("LLM_ENDPOINT_NAME")
    uc_tool_names = os.environ.get("UC_TOOL_NAMES", "")
    vs_name = os.environ.get("VS_NAME", "")
    
    # 必須項目の検証
    if not llm_endpoint:
        raise ValueError("LLM_ENDPOINT_NAME environment variable is required")
    
    if not uc_tool_names:
        raise ValueError("UC_TOOL_NAMES environment variable is required")

    if not vs_name:
        raise ValueError("VS_NAME environment variable is required")

    # ツール名を分割（空文字列や空白を除去）
    tool_names = [name.strip() for name in uc_tool_names.split(",") if name.strip()]

    config = {
        "llm_endpoint": llm_endpoint,
        "uc_tool_names": tool_names,
        "vs_name": vs_name
    }
    
    return config

# 設定を取得
config = get_env_config()

LLM_ENDPOINT_NAME = config['llm_endpoint']
UC_TOOL_NAMES = config['uc_tool_names']
VS_NAME = config['vs_name']

# 設定確認用の出力
print("エージェントの設定:")
print(f"LLM_ENDPOINT_NAME: {LLM_ENDPOINT_NAME}")
print(f"UC_TOOL_NAMES: {UC_TOOL_NAMES}")
print(f"VS_NAME: {VS_NAME}")

# LangChain/MLflowの自動ロギングを有効化
mlflow.langchain.autolog()

# Databricks Function Clientを初期化し、UC関数クライアントとしてセット
client = DatabricksFunctionClient(disable_notice=True, suppress_warnings=True)
set_uc_function_client(client)

############################################
# LLMインスタンスの作成
############################################
llm = ChatDatabricks(
    endpoint=config["llm_endpoint"]
)

# システムプロンプト（エージェントの振る舞いを制御）
#system_prompt = "あなたはDatabricksラボのカスタマーサクセススペシャリストです。ユーザーからの製品に関する質問に対し、必要な情報はツールを使って取得し、ユーザーが製品を十分に理解できるようサポートしてください。お客様の興味を引くであろう情報を可能な限り盛り込んで、すべてのやり取りで価値を提供することを心がけてください。"

system_prompt = "あなたはDatabricksラボのカスタマーサクセススペシャリストです。ユーザーからの製品に関する質問に対し、必要な情報はツールを使って取得し、質問に対してのみ簡潔に答え、架空の機能や色、一般的なコメントは加えないでください。マーケティング的な表現や余計な背景説明は不要です。"


###############################################################################
## エージェント用のツールを定義。これにより、テキスト生成以外のデータ取得やアクションが可能になる
## さらに多くのツールの作成や使用例については
## https://docs.databricks.com/generative-ai/agent-framework/agent-tool.html を参照
###############################################################################

############################################
# ツールの作成
############################################
def create_tools() -> List[BaseTool]:
    """環境変数の設定に基づいてツールを作成"""
    tools = []
    
    # Vector Searchツールの追加
    if VS_NAME:
        try:            
            vs_tool = VectorSearchRetrieverTool(
                index_name=VS_NAME,
                tool_name="search_product_docs",
                num_results=3, # VSから3件の文書を取得
                #num_results=1, # VSから1件の文書を取得
                tool_description="このツールを使用して製品ドキュメントを検索します。"
            )
            tools.append(vs_tool)
            print(f"Vector Searchツールを追加: {VS_NAME}")
        except Exception as e:
            print(f"Warning: Vector Searchツール {VS_NAME} をロードできませんでした: {e}")
    
    # UC関数ツールの追加
    if UC_TOOL_NAMES:
        try:
            uc_toolkit = UCFunctionToolkit(function_names=UC_TOOL_NAMES)
            tools.extend(uc_toolkit.tools)
            print(f"UC関数ツールを追加: {UC_TOOL_NAMES}")
        except Exception as e:
            print(f"Warning: UCツール {UC_TOOL_NAMES} を追加できませんでした: {e}")
    
    return tools

# ツールを作成
# MLflowのロギングでも使用
tools = create_tools()

#####################
## エージェントのロジックを定義
#####################
def create_tool_calling_agent(
    model: LanguageModelLike,
    tools: Union[Sequence[BaseTool], ToolNode],
    system_prompt: Optional[str] = None,
) -> CompiledGraph:
    # モデルにツールをバインド
    model = model.bind_tools(tools)

    # 次にどのノードに進むかを決定する関数を定義
    def should_continue(state: ChatAgentState):
        messages = state["messages"]
        last_message = messages[-1]
        # 関数呼び出しがあれば継続、なければ終了
        if last_message.get("tool_calls"):
            return "continue"
        else:
            return "end"

    # システムプロンプトを先頭に付与する前処理
    if system_prompt:
        preprocessor = RunnableLambda(
            lambda state: [{"role": "system", "content": system_prompt}]
            + state["messages"]
        )
    else:
        preprocessor = RunnableLambda(lambda state: state["messages"])
    model_runnable = preprocessor | model

    # モデル呼び出し用の関数
    def call_model(
        state: ChatAgentState,
        config: RunnableConfig,
    ):
        response = model_runnable.invoke(state, config)

        return {"messages": [response]}

    # カスタムツール実行関数
    def execute_tools(state: ChatAgentState):
        messages = state["messages"]
        last_message = messages[-1]
        
        # tool_callsを取得
        tool_calls = last_message.get("tool_calls", [])
        if not tool_calls:
            return {"messages": []}
        
        # ツールを実行
        tool_outputs = []
        for tool_call in tool_calls:
            tool_name = tool_call.get("function", {}).get("name") if isinstance(tool_call, dict) else tool_call.function.name
            tool_args = tool_call.get("function", {}).get("arguments") if isinstance(tool_call, dict) else tool_call.function.arguments
            tool_id = tool_call.get("id") if isinstance(tool_call, dict) else tool_call.id
            
            # ツールを見つけて実行
            tool_result = None
            for tool in tools:
                if tool.name == tool_name:
                    try:
                        # 引数をパース
                        import json
                        if isinstance(tool_args, str):
                            args = json.loads(tool_args)
                        else:
                            args = tool_args
                        
                        # ツールを実行
                        result = tool.invoke(args)
                        tool_result = str(result)
                    except Exception as e:
                        tool_result = f"Error executing tool: {str(e)}"
                    break
            
            if tool_result is None:
                tool_result = f"Tool {tool_name} not found"
            
            # ツール実行結果のメッセージを作成
            tool_message = {
                "role": "tool",
                "content": tool_result,
                "tool_call_id": tool_id,
                "name": tool_name
            }
            tool_outputs.append(tool_message)
        
        return {"messages": tool_outputs}

    # LangGraphのワークフローを構築
    workflow = StateGraph(ChatAgentState)

    workflow.add_node("agent", RunnableLambda(call_model))
    workflow.add_node("tools", execute_tools)

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()

# LangGraphChatAgentクラス（MLflow推論用ラッパー）
class LangGraphChatAgent(ChatAgent):
    def __init__(self, agent: CompiledStateGraph):
        self.agent = agent

    def _convert_messages_to_dict(self, messages: list[ChatAgentMessage]) -> list[dict]:
        """ChatAgentMessageを辞書形式に変換（修正版）"""
        converted = []
        
        # messagesがNoneまたは空の場合の処理
        if not messages:
            return converted
            
        for msg in messages:
            try:
                if msg is None:
                    print("Warning: None message encountered")
                    continue
                    
                # ChatAgentMessageオブジェクトを辞書に変換
                if hasattr(msg, 'dict'):
                    msg_dict = msg.dict()
                elif isinstance(msg, dict):
                    msg_dict = msg
                else:
                    print(f"Warning: Unexpected message type: {type(msg)}")
                    continue
                
                # toolロールのメッセージの場合、contentが空の場合は処理
                if msg_dict.get("role") == "tool":
                    # contentが空またはNoneの場合、デフォルト値を設定
                    if not msg_dict.get("content"):
                        msg_dict["content"] = "Tool execution completed"
                    
                    # tool_call_idが必要な場合は設定
                    if "tool_call_id" not in msg_dict and msg_dict.get("id"):
                        msg_dict["tool_call_id"] = msg_dict["id"]
                
                converted.append(msg_dict)
            except Exception as e:
                print(f"Error converting message: {e}, Message: {msg}")
                continue
        
        return converted

    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        # 入力メッセージを辞書形式に変換
        request = {"messages": self._convert_messages_to_dict(messages)}

        messages = []
        # LangGraphのストリームからメッセージを収集（元のコードに近い形で）
        try:
            for event in self.agent.stream(request, stream_mode="updates"):
                if event and isinstance(event, dict):
                    for node_data in event.values():
                        if node_data and isinstance(node_data, dict) and "messages" in node_data:
                            for msg in node_data.get("messages", []):
                                if msg is None:
                                    continue
                                    
                                # メッセージオブジェクトを辞書に変換
                                if hasattr(msg, 'dict'):
                                    msg_dict = msg.dict()
                                elif isinstance(msg, dict):
                                    msg_dict = msg
                                else:
                                    print(f"Warning: Unexpected message type: {type(msg)}")
                                    continue
                                
                                # toolメッセージの内容を確認
                                if msg_dict.get("role") == "tool":
                                    # contentがない場合はデフォルト値を設定
                                    if not msg_dict.get("content") and not msg_dict.get("tool_calls"):
                                        msg_dict["content"] = "Tool executed successfully"
                                    
                                    # tool_call_idが必要な場合
                                    if "tool_call_id" not in msg_dict and "id" in msg_dict:
                                        msg_dict["tool_call_id"] = msg_dict["id"]
                                
                                try:
                                    messages.append(ChatAgentMessage(**msg_dict))
                                except Exception as e:
                                    print(f"Warning: Failed to create ChatAgentMessage: {e}")
                                    print(f"Message data: {msg_dict}")
                                    
        except Exception as e:
            print(f"Error in predict method: {e}")
            import traceback
            traceback.print_exc()
            
        return ChatAgentResponse(messages=messages)

    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        # 入力メッセージを辞書形式に変換
        request = {"messages": self._convert_messages_to_dict(messages)}
        
        # ストリームで逐次応答を生成
        try:
            for event in self.agent.stream(request, stream_mode="updates"):
                if event and isinstance(event, dict):
                    for node_data in event.values():
                        if node_data and isinstance(node_data, dict) and "messages" in node_data:
                            for msg in node_data.get("messages", []):
                                if msg is None:
                                    continue
                                    
                                # メッセージオブジェクトを辞書に変換
                                if hasattr(msg, 'dict'):
                                    msg_dict = msg.dict()
                                elif isinstance(msg, dict):
                                    msg_dict = msg
                                else:
                                    print(f"Warning: Unexpected message type in stream: {type(msg)}")
                                    continue
                                
                                # toolメッセージの内容を確認
                                if msg_dict.get("role") == "tool":
                                    # contentがない場合はデフォルト値を設定
                                    if not msg_dict.get("content") and not msg_dict.get("tool_calls"):
                                        msg_dict["content"] = "Tool executed successfully"
                                    
                                    # tool_call_idが必要な場合
                                    if "tool_call_id" not in msg_dict and "id" in msg_dict:
                                        msg_dict["tool_call_id"] = msg_dict["id"]
                                
                                try:
                                    yield ChatAgentChunk(**{"delta": msg_dict})
                                except Exception as e:
                                    print(f"Warning: Failed to create ChatAgentChunk: {e}")
                                    continue
                                    
        except Exception as e:
            print(f"Error in predict_stream method: {e}")
            import traceback
            traceback.print_exc()
            return

# エージェントオブジェクトを作成し、mlflow.models.set_model()で推論時に使用するエージェントとして指定
agent = create_tool_calling_agent(llm, tools, system_prompt)
AGENT = LangGraphChatAgent(agent)
mlflow.models.set_model(AGENT)