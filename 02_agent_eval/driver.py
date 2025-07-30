# Databricks notebook source
# MAGIC %md
# MAGIC サーバレス

# COMMAND ----------

# MAGIC %md
# MAGIC # ハンズオンラボ：Databricksでエージェントシステムを構築する
# MAGIC
# MAGIC ## パート2 - エージェント評価
# MAGIC エージェントを作成したので、そのパフォーマンスをどのように評価するのでしょうか？
# MAGIC 第2部では、評価に焦点を当てるために製品サポートエージェントを作成します。
# MAGIC このエージェントは、RAGアプローチを使用して製品ドキュメントを活用し、製品に関する質問に回答します。
# MAGIC
# MAGIC ### 2.1 新しいエージェントとリトリーバーツールの定義
# MAGIC - [**agent.py**]($./agent.py)：サンプルエージェントが設定されています。まずこのファイルを確認し、構成要素を理解しましょう
# MAGIC - **ベクター検索**：特定の製品に関連するドキュメントを検索できるベクター検索エンドポイントを作成しました。
# MAGIC - **リトリーバー関数の作成**：リトリーバーのプロパティを定義し、LLMから呼び出せるようにパッケージ化します。
# MAGIC
# MAGIC ### 2.2 評価データセットの作成
# MAGIC - サンプルの評価データセットを用意していますが、[合成的に生成](https://www.databricks.com/blog/streamline-ai-agent-evaluation-with-new-synthetic-data-capabilities)することも可能です。
# MAGIC
# MAGIC ### 2.3 MLflow.evaluate() の実行
# MAGIC - MLflowは評価データセットを使ってエージェントの応答をテストします
# MAGIC - LLMジャッジが出力をスコア化し、すべてを見やすいUIでまとめます
# MAGIC
# MAGIC ### 2.4 必要な改善を行い、再評価を実施
# MAGIC - 評価結果からフィードバックを得てリトリーバー設定を変更
# MAGIC - 再度評価を実施し、改善を確認しましょう！

# COMMAND ----------

# MAGIC %pip install -U -qqqq mlflow-skinny[databricks] langgraph==0.3.4 databricks-langchain databricks-agents uv
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,エージェントが動作することを確認するためのクイックなテスト
from agent import AGENT

AGENT.predict({"messages": [{"role": "user", "content": "Soundwave X5 Pro ヘッドフォンのトラブルシューティングのコツを教えてください。"}]})

# COMMAND ----------

# MAGIC %md
# MAGIC ### `agent` をMLflowモデルとしてログに記録する
# MAGIC [agent]($./agent)ノートブックのコードとしてエージェントをログに記録します。詳細は[MLflow - コードからのモデル](https://mlflow.org/docs/latest/models.html#models-from-code)を参照してください。

# COMMAND ----------

# デプロイ時に自動認証パススルーを指定するためのDatabricksリソースを決定
import mlflow
from agent import tools, LLM_ENDPOINT_NAME
from databricks_langchain import VectorSearchRetrieverTool
from mlflow.models.resources import DatabricksFunction, DatabricksServingEndpoint
from unitycatalog.ai.langchain.toolkit import UnityCatalogTool

resources = [DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT_NAME)]
for tool in tools:
    if isinstance(tool, VectorSearchRetrieverTool):
        resources.extend(tool.resources)
    elif isinstance(tool, UnityCatalogTool):
        resources.append(DatabricksFunction(function_name=tool.uc_function_name))

input_example = {
    "messages": [
        {
            "role": "user",
            "content": "Aria Modern Bookshelfの利用可能な色オプションは何ですか？"
        }
    ]
}

with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model="agent.py",
        input_example=input_example,
        resources=resources,
        extra_pip_requirements=[
            "databricks-connect"
        ]
    )

# COMMAND ----------

# モデルをロードし、予測関数を作成
logged_model_uri = f"runs:/{logged_agent_info.run_id}/agent"
loaded_model = mlflow.pyfunc.load_model(logged_model_uri)

def predict_wrapper(query):
    # チャット形式モデル用の入力を整形
    model_input = {
        "messages": [{"role": "user", "content": query}]
    }
    response = loaded_model.predict(model_input)
    
    messages = response['messages']
    return messages[-1]['content']

# COMMAND ----------

# MAGIC %md
# MAGIC ## エージェントを[エージェント評価](https://docs.databricks.com/generative-ai/agent-evaluation/index.html)で評価する
# MAGIC
# MAGIC 評価データセットのリクエストや期待される応答を編集し、エージェントを反復しながら評価を実行し、mlflowを活用して計算された品質指標を追跡できます。

# COMMAND ----------

import pandas as pd

data = {
    "request": [
        "Aria Modern Bookshelfの利用可能な色オプションは何ですか？",
        "Aurora Oak Coffee Tableを傷つけずに掃除するにはどうすればよいですか？",
        "BlendMaster Elite 4000は使用後にどのように掃除すればよいですか？",
        "Flexi-Comfort Office Deskは何色展開ですか？",
        "StormShield Pro メンズ防水ジャケットのサイズ展開は？"
    ],
    "expected_facts": [
        [
            "Aria Modern Bookshelfはナチュラルオーク仕上げで利用可能です。",
            "Aria Modern Bookshelfはブラック仕上げで利用可能です。",
            "Aria Modern Bookshelfはホワイト仕上げで利用可能です。"
        ],
        [
            "柔らかく少し湿らせた布で掃除してください。",
            "研磨剤入りのクリーナーは使用しないでください。"
        ],
        [
            "BlendMaster Elite 4000のジャーはすすいでください。",
            "ぬるま湯ですすいでください。",
            "使用後は毎回掃除してください。"
        ],
        [
            "Flexi-Comfort Office Deskは3色展開です。"
        ],
        [
            "StormShield Pro メンズ防水ジャケットのサイズはS、M、L、XL、XXLです。"
        ]
    ]
}

eval_dataset = pd.DataFrame(data)

# COMMAND ----------

from mlflow.genai.scorers import Guidelines, Safety
import mlflow.genai

# 評価用データセットを作成
eval_data = []
for request, facts in zip(data["request"], data["expected_facts"]):
    eval_data.append({
        "inputs": {
            "query": request  # 関数の引数と一致させる
        },
        "expected_response": "\n".join(facts)
    })

# 評価用スコアラーを定義
# LLMジャッジが応答を評価するためのガイドライン

# 製品情報評価に特化したカスタムスコアラーを定義
scorers = [
    Guidelines(
        guidelines="""応答にはすべての期待される事実が含まれている必要があります:
        - 関連する場合はすべての色やサイズを列挙する（部分的なリストは不可）
        - 関連する場合は正確な仕様を記載する（例:「5 ATM」など曖昧な表現は不可）
        - 掃除手順を尋ねられた場合はすべての手順を含める
        いずれかの事実が欠落または誤っている場合は不合格とする。""",
        name="completeness_and_accuracy",
    ),
    Guidelines(
        guidelines="""応答は明確かつ直接的でなければなりません:
        - 質問に正確に答える
        - 選択肢はリスト形式、手順はステップ形式で記載
        - マーケティング的な表現や余計な背景説明は不要
        - 簡潔かつ完全であること。""",
        name="relevance_and_structure",
    ),
    Guidelines(
        guidelines="""応答は話題から逸脱しないこと:
        - 質問された製品のみについて回答する
        - 架空の機能や色を追加しない
        - 一般的なアドバイスは含めない
        - リクエストに記載された製品名を正確に使用すること。""",
        name="product_specificity",
    ),
]

# COMMAND ----------

print("評価を実行中...")
with mlflow.start_run():
    results = mlflow.genai.evaluate(
        data=eval_data,
        predict_fn=predict_wrapper, 
        scorers=scorers,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## [agent.py]($./agent.py) ファイルに戻り、プロンプトを変更してマーケティングの誇張を減らしましょう。

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルをUnity Catalogに登録する
# MAGIC
# MAGIC 以下の `catalog`、`schema`、`model_name` を更新して、MLflowモデルをUnity Catalogに登録します。

# COMMAND ----------

from databricks.sdk import WorkspaceClient
import os

mlflow.set_registry_uri("databricks-uc")

# ワークスペースクライアントを使用して現在のユーザー情報を取得
w = WorkspaceClient()
user_email = w.current_user.me().display_name
username = user_email.split("@")[0]

# カタログとスキーマはラボ環境で自動作成済み
#catalog_name = f"{username}"
#schema_name = "agents"
catalog_name = "takaakiyayoi_catalog"
schema_name = "agents_lab"

# UCモデル用のカタログ、スキーマ、モデル名を定義
model_name = "product_agent"
UC_MODEL_NAME = f"{catalog_name}.{schema_name}.{model_name}"

# モデルをUCに登録
uc_registered_model_info = mlflow.register_model(model_uri=logged_agent_info.model_uri, name=UC_MODEL_NAME)

# COMMAND ----------

from IPython.display import display, HTML

# DatabricksのホストURLを取得
workspace_url = spark.conf.get('spark.databricks.workspaceUrl')

# 作成したエージェントへのHTMLリンクを作成
html_link = f'<a href="https://{workspace_url}/explore/data/models/{catalog_name}/{schema_name}/product_agent" target="_blank">登録済みエージェントをUnity Catalogで表示</a>'
display(HTML(html_link))

# COMMAND ----------

# MAGIC %md
# MAGIC ## エージェントのデプロイ
# MAGIC
# MAGIC ##### 注意: これはラボユーザーには無効ですが、自分のワークスペースでは機能します

# COMMAND ----------

from databricks import agents

# モデルをレビューアプリおよびモデルサービングエンドポイントにデプロイ

# ラボ環境では無効化されていますが、すでにエージェントはデプロイ済みです！
#agents.deploy(UC_MODEL_NAME, uc_registered_model_info.version, tags = {"endpointSource": "Agent Lab"})
