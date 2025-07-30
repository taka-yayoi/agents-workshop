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
from mlflow.langchain.chat_agent_langgraph import ChatAgentState, ChatAgentToolNode
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)

mlflow.langchain.autolog()

client = DatabricksFunctionClient(disable_notice=True)
set_uc_function_client(client)

############################################
# LLMエンドポイントとシステムプロンプトを定義
############################################
LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"
llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)

system_prompt = "ツールを使用して必要な情報をすべて取得し、丁寧に応答してください。ツールの呼び出し結果のみを返さないでください。"

###############################################################################
## エージェント用のツールを定義。これにより、テキスト生成以外のデータ取得やアクションが可能になる
## さらに多くのツールの作成や使用例については
## https://docs.databricks.com/generative-ai/agent-framework/agent-tool.html を参照
###############################################################################
tools = []

# Unity CatalogのUDFをエージェントツールとして利用可能
uc_tool_names = ["takaakiyayoi_catalog.agents_lab.*"]
uc_toolkit = UCFunctionToolkit(function_names=uc_tool_names)
tools.extend(uc_toolkit.tools)

# # Databricksベクトル検索インデックスをツールとして利用
# # 詳細は https://docs.databricks.com/generative-ai/agent-framework/unstructured-retrieval-tools.html を参照

# ベクトル検索インデックスから情報を取得するツールを追加
# num_resultsは返される結果数を決定する重要な入力
retriever_tool = VectorSearchRetrieverTool(
  #index_name="takaakiyayoi_catalog.agents_lab.product_docs_index_jpn",
  index_name="takaakiyayoi_catalog.agents_lab.product_docs_index",
  tool_name="search_product_docs",
  tool_description="このツールを使用して製品ドキュメントを検索します。",
  num_results=3,
  disable_notice=True
)

tools.append(retriever_tool)

#####################
## エージェントのロジックを定義
#####################

def create_tool_calling_agent(
    model: LanguageModelLike,
    tools: Union[Sequence[BaseTool], ToolNode],
    system_prompt: Optional[str] = None,
) -> CompiledGraph:
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

    if system_prompt:
        preprocessor = RunnableLambda(
            lambda state: [{"role": "system", "content": system_prompt}]
            + state["messages"]
        )
    else:
        preprocessor = RunnableLambda(lambda state: state["messages"])
    model_runnable = preprocessor | model

    def call_model(
        state: ChatAgentState,
        config: RunnableConfig,
    ):
        response = model_runnable.invoke(state, config)

        return {"messages": [response]}

    workflow = StateGraph(ChatAgentState)

    workflow.add_node("agent", RunnableLambda(call_model))
    workflow.add_node("tools", ChatAgentToolNode(tools))

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


class LangGraphChatAgent(ChatAgent):
    def __init__(self, agent: CompiledStateGraph):
        self.agent = agent

    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        request = {"messages": self._convert_messages_to_dict(messages)}

        messages = []
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                messages.extend(
                    ChatAgentMessage(**msg) for msg in node_data.get("messages", [])
                )
        return ChatAgentResponse(messages=messages)

    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        request = {"messages": self._convert_messages_to_dict(messages)}
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                yield from (
                    ChatAgentChunk(**{"delta": msg}) for msg in node_data["messages"]
                )

# エージェントオブジェクトを作成し、mlflow.models.set_model()で推論時に使用するエージェントとして指定
agent = create_tool_calling_agent(llm, tools, system_prompt)
AGENT = LangGraphChatAgent(agent)
mlflow.models.set_model(AGENT)