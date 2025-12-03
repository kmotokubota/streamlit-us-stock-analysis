-- ============================================================================
-- Snowflake 株式分析アプリ セットアップスクリプト
-- ============================================================================
-- このスクリプトを実行すると、GitHub連携とStreamlit in Snowflakeが自動で作成されます。
-- 
-- 前提条件:
-- 1. ACCOUNTADMINロールへのアクセス権限
-- 2. COMPUTE_WHウェアハウスが存在すること
-- 3. Snowflake Marketplaceから「Cybersyn Financial & Economic Essentials」を取得済み
-- ============================================================================

-- ロールとウェアハウスの設定
USE ROLE ACCOUNTADMIN;
USE WAREHOUSE COMPUTE_WH;

-- ============================================================================
-- データベースとスキーマの作成
-- ============================================================================
CREATE OR REPLACE DATABASE STOCK_ANALYSIS;
CREATE OR REPLACE SCHEMA STOCK_ANALYSIS.STOCK_ANALYSIS_SCHEMA;

USE DATABASE STOCK_ANALYSIS;
USE SCHEMA STOCK_ANALYSIS_SCHEMA;

-- ============================================================================
-- Cortexクロスリージョン設定（必要に応じて）
-- ============================================================================
ALTER ACCOUNT SET CORTEX_ENABLED_CROSS_REGION = 'ANY_REGION';

-- ============================================================================
-- GitHub API統合の作成
-- ============================================================================
-- GitHubリポジトリとの連携を可能にするAPI統合を作成
CREATE OR REPLACE API INTEGRATION GIT_API_INTEGRATION
    API_PROVIDER = git_https_api
    API_ALLOWED_PREFIXES = ('https://github.com/kmotokubota/')
    ENABLED = TRUE;

-- ============================================================================
-- Gitリポジトリの作成
-- ============================================================================
-- GitHubリポジトリをSnowflakeに登録
CREATE OR REPLACE GIT REPOSITORY GIT_REPO_STOCK_ANALYSIS
    API_INTEGRATION = GIT_API_INTEGRATION
    ORIGIN = 'https://github.com/kmotokubota/streamlit-us-stock-analysis.git';

-- ============================================================================
-- Streamlit in Snowflakeの作成
-- ============================================================================
-- GitHubから直接Streamlitアプリを作成
CREATE OR REPLACE STREAMLIT STOCK_PRICE_ANALYSIS_APP
    FROM @GIT_REPO_STOCK_ANALYSIS/branches/main/
    MAIN_FILE = 'stock_price_analysis_app.py'
    QUERY_WAREHOUSE = COMPUTE_WH;

-- ============================================================================
-- セットアップ完了メッセージ
-- ============================================================================
SELECT '✅ セットアップが完了しました！' AS STATUS,
       'Snowsightの「Streamlit」タブから STOCK_PRICE_ANALYSIS_APP を開いてください。' AS NEXT_STEP;

