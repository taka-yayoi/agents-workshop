# Databricks notebook source
# MAGIC %md
# MAGIC # ハンズオンラボ：Databricksでエージェントシステムを構築する
# MAGIC
# MAGIC ## パート1 - 最初のエージェントを設計する
# MAGIC この最初のエージェントは、カスタマーサービス担当者のワークフローに従い、さまざまなエージェント機能を説明します。
# MAGIC 商品返品の処理に焦点を当て、具体的な手順を追っていきます。
# MAGIC
# MAGIC ### 1.1 シンプルなツールを作成する
# MAGIC - **SQL関数**：返品処理ワークフローの各ステップで重要となるデータへアクセスするクエリを作成します。
# MAGIC - **シンプルなPython関数**：言語モデルの一般的な制限を克服するためのPython関数を作成し、登録します。
# MAGIC
# MAGIC ### 1.2 LLMとの統合【AI Playground】
# MAGIC - 作成したツールを言語モデル（LLM）と組み合わせてAI Playgroundで利用します。
# MAGIC
# MAGIC ### 1.3 エージェントのテスト【AI Playground】
# MAGIC - エージェントに質問し、応答を観察します。
# MAGIC - MLflowトレースを活用してエージェントのパフォーマンスをさらに深く探ります。

# COMMAND ----------

# ウィジェットをクリア
dbutils.widgets.removeAll()

# COMMAND ----------

# DBTITLE 1,ライブラリのインストール
# MAGIC %pip install -qqqq -U -r requirements.txt
# MAGIC # パッケージをPython環境にロードするために再起動します
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../config

# COMMAND ----------

# DBTITLE 1,パラメーター設定
from databricks.sdk import WorkspaceClient
import yaml
import os
import re

# ワークスペースクライアントを使用して現在のユーザーに関する情報を取得
w = WorkspaceClient()
user_email = w.current_user.me().emails[0].value
username = user_email.split('@')[0]
username = re.sub(r'[^a-zA-Z0-9_]', '_', username) # 特殊文字をアンダースコアに置換

# スキーマを指定します
user_schema_name = f"agents_lab_{username}" # ユーザーごとのスキーマ

print("あなたのカタログは:", catalog_name)
print("あなたのスキーマは:", user_schema_name)

# SQL/Python関数を作成する際にこれらの値を参照できるようにします
dbutils.widgets.text("catalog_name", defaultValue=catalog_name, label="Catalog Name")
dbutils.widgets.text("system_schema_name", defaultValue=system_schema_name, label="System Schema Name")
dbutils.widgets.text("user_schema_name", defaultValue=user_schema_name, label="User Schema Name")

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog_name}.{user_schema_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC # カスタマーサービス返品処理ワークフロー
# MAGIC
# MAGIC 以下は、カスタマーサービス担当者が**返品を処理する際**に通常従う**主要なステップ**の構造化された概要です。このワークフローにより、サポートチーム全体で一貫性と明確さが確保されます。
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 1. 処理キュー内の最新の返品を取得する
# MAGIC - **アクション**：チケッティングまたは返品システムから最新の返品リクエストを特定し、取得します。  
# MAGIC - **理由**：最も緊急または次に対応すべき顧客の問題に取り組んでいることを保証します。
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,処理キューにおける最新の返品を取得
# MAGIC %sql
# MAGIC -- インタラクションの日付、問題のカテゴリ、問題の説明、および顧客の名前を選択
# MAGIC SELECT 
# MAGIC   cast(date_time as date) as case_time, 
# MAGIC   issue_category, 
# MAGIC   issue_description, 
# MAGIC   name
# MAGIC FROM IDENTIFIER(:catalog_name || '.' || :system_schema_name || '.cust_service_data')
# MAGIC -- インタラクションの日付と時刻で結果を降順に並べ替え
# MAGIC ORDER BY date_time DESC
# MAGIC -- 結果を最新のインタラクションに制限
# MAGIC LIMIT 1

# COMMAND ----------

# MAGIC %sql
# MAGIC USE CATALOG IDENTIFIER(:catalog_name);
# MAGIC USE SCHEMA IDENTIFIER(:system_schema_name)

# COMMAND ----------

# DBTITLE 1,Unity Catalogに登録される関数を作成
# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION 
# MAGIC   IDENTIFIER(:catalog_name || '.' || :user_schema_name || '.get_latest_return')()
# MAGIC RETURNS TABLE(purchase_date DATE, issue_category STRING, issue_description STRING, name STRING)
# MAGIC COMMENT '最新のカスタマーサービス対応（返品など）を返します。'
# MAGIC RETURN (
# MAGIC   SELECT 
# MAGIC     CAST(date_time AS DATE) AS purchase_date,
# MAGIC     issue_category,
# MAGIC     issue_description,
# MAGIC     name
# MAGIC   FROM cust_service_data
# MAGIC   ORDER BY date_time DESC
# MAGIC   LIMIT 1
# MAGIC );

# COMMAND ----------

# DBTITLE 1,最新の返品を取得するために関数呼び出しをテスト
# MAGIC %sql
# MAGIC select * from IDENTIFIER(:catalog_name || '.' || :user_schema_name || '.get_latest_return')()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## 2. 会社のポリシーを取得する
# MAGIC - **アクション**：返品、返金、および交換に関する内部ナレッジベースまたはポリシー文書にアクセスします。  
# MAGIC - **理由**：会社のガイドラインに準拠していることを確認することで、潜在的なエラーや対立を防ぎます。
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,返品ポリシーを取得する関数の呼び出し
# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION
# MAGIC   IDENTIFIER(:catalog_name || '.' || :user_schema_name || '.get_return_policy')()
# MAGIC RETURNS TABLE (
# MAGIC   policy           STRING,
# MAGIC   policy_details   STRING,
# MAGIC   last_updated     DATE
# MAGIC )
# MAGIC COMMENT '返品ポリシーの詳細を返します'
# MAGIC LANGUAGE SQL
# MAGIC RETURN (
# MAGIC   SELECT
# MAGIC     policy,
# MAGIC     policy_details,
# MAGIC     last_updated
# MAGIC   FROM policies
# MAGIC   WHERE policy = 'Return Policy'
# MAGIC   LIMIT 1
# MAGIC );

# COMMAND ----------

# DBTITLE 1,返品ポリシーを取得する関数のテスト
# MAGIC %sql
# MAGIC select * from IDENTIFIER(:catalog_name || '.' || :user_schema_name || '.get_return_policy')()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## 3. 最新の返品のUserIDを取得する
# MAGIC - **アクション**：返品リクエストの詳細からユーザーの一意の識別子を記録します。  
# MAGIC - **理由**：正しいユーザーデータを正確に参照することで、処理が効率化され、顧客記録の混同を防ぎます。
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,名前に基づいてuserIDを取得する関数の作成
# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION
# MAGIC   IDENTIFIER(:catalog_name || '.' || :user_schema_name || '.get_user_id')(user_name STRING)
# MAGIC RETURNS STRING
# MAGIC COMMENT 'これは顧客の名前を入力として受け取り、対応するユーザーIDを返します'
# MAGIC LANGUAGE SQL
# MAGIC RETURN 
# MAGIC SELECT customer_id 
# MAGIC FROM cust_service_data 
# MAGIC WHERE name = user_name
# MAGIC LIMIT 1
# MAGIC ;

# COMMAND ----------

# DBTITLE 1,名前に基づいてuserIDを取得する関数のテスト
# MAGIC %sql
# MAGIC
# MAGIC --新たなパラメータ構文 (MLR > 15.1)
# MAGIC select IDENTIFIER(:catalog_name || '.' || :user_schema_name || '.get_user_id')('Nicolas Pelaez');

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## 4. UserIDを使って注文履歴を照会する
# MAGIC - **アクション**：UserIDを使って注文管理システムや顧客データベースを検索します。  
# MAGIC - **理由**：過去の購入履歴や返品傾向、特記事項を確認することで、次に取るべき適切な対応（例：返品の適格性の確認）を判断できます。
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,userIDに基づいて注文履歴を取得する関数の作成
# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION
# MAGIC   IDENTIFIER(:catalog_name || '.' || :user_schema_name || '.get_order_history')(user_id STRING)
# MAGIC RETURNS TABLE (returns_last_12_months INT, issue_category STRING)
# MAGIC COMMENT 'これは顧客のuser_idを入力として受け取り、過去12か月間の返品数と問題カテゴリを返します'
# MAGIC LANGUAGE SQL
# MAGIC RETURN 
# MAGIC SELECT count(*) as returns_last_12_months, issue_category 
# MAGIC FROM cust_service_data 
# MAGIC WHERE customer_id = user_id 
# MAGIC GROUP BY issue_category;

# COMMAND ----------

# DBTITLE 1,userIDに基づいて注文履歴を取得する関数のテスト
# MAGIC %sql
# MAGIC select * from IDENTIFIER(:catalog_name || '.' || :user_schema_name || '.get_order_history')('453e50e0-232e-44ea-9fe3-28d550be6294')

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## 5. LLMに今日の日付を取得するPython関数を提供する
# MAGIC - **アクション**：LLM（大規模言語モデル）に現在の日付を提供できる**Python関数**を用意します。  
# MAGIC - **理由**：日付の自動取得により、集荷のスケジューリング、返金のタイムライン、連絡期限などの管理が容易になります。
# MAGIC
# MAGIC ###### 注：System.ai.python_execに登録された関数を使うことで、LLMが生成したコードをサンドボックス環境で実行できます
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,非常にシンプルなPython関数
def get_todays_date() -> str:
    """
    今日の日付を 'YYYY-MM-DD' 形式で返します。

    Returns:
        str: 今日の日付を 'YYYY-MM-DD' 形式で返します。
    """
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d")

# COMMAND ----------

# DBTITLE 1,Python関数のテスト
today = get_todays_date()
today

# COMMAND ----------

# DBTITLE 1,Python関数をUnity Catalogに登録
from unitycatalog.ai.core.databricks import DatabricksFunctionClient

client = DatabricksFunctionClient()

# このツールをUCにデプロイし、ツールのdocstringと型ヒントに基づいてUCのメタデータを自動的に設定します
python_tool_uc_info = client.create_python_function(func=get_todays_date, catalog=catalog_name, schema=user_schema_name, replace=True)

# ツールはUC内の `{catalog}.{schema}.{func}` という名前の関数にデプロイされます。ここで {func} は関数の名前です
# デプロイされたUnity Catalog関数名を表示します
display(f"デプロイされたUnity Catalog関数名: {python_tool_uc_info.full_name}")

# COMMAND ----------

# DBTITLE 1,作成した関数を見てみましょう
from IPython.display import display, HTML

# DatabricksのホストURLを取得
workspace_url = spark.conf.get('spark.databricks.workspaceUrl')

# 作成した関数へのHTMLリンクを作成
html_link = f'<a href="https://{workspace_url}/explore/data/functions/{catalog_name}/{user_schema_name}/get_todays_date" target="_blank">登録済み関数をUnity Catalogで確認</a>'
display(HTML(html_link))

# COMMAND ----------

# MAGIC %md
# MAGIC ## では、これらの関数を使用して最初のエージェントを組み立てる方法をAIプレイグラウンドで見てみましょう！
# MAGIC
# MAGIC - **システムプロンプト**：`すべての社内ポリシーが満たされていると確信できるまで、ツールを呼び出すこと`
# MAGIC - **質問例**：`当社のポリシーに基づいて、最新の返品を受け付けるべきでしょうか？`
# MAGIC
# MAGIC ### AIプレイグラウンドは、左のナビゲーションバーの「AI/ML」から見つけることができます。または、以下に作成されたリンクを使用することもできます。

# COMMAND ----------

# DBTITLE 1,AI Playgroundへのリンクを作成
# AI PlaygroundへのHTMLリンクを作成
html_link = f'<a href="https://{workspace_url}/ml/playground" target="_blank">AI Playgroundへ移動</a>'
display(HTML(html_link))
