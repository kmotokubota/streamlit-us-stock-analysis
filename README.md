# 📈 Snowflake マーケットプレイス 株式分析アプリ

SnowflakeのマーケットプレイスデータとStreamlitを使用した、インタラクティブな株式分析アプリケーションです。

## 🚀 機能

### 主要機能
- **銘柄選択**: 人気銘柄から選択 or ティッカーコードを直接入力
- **カスタム移動平均**: 5日、25日、50日、200日移動平均 + カスタム期間設定
- **テクニカル指標**: RSI、ボリンジャーバンド、出来高表示
- **分析期間**: 1ヶ月〜2年の期間設定
- **データダウンロード**: CSV形式でデータをダウンロード

### インタラクティブ機能
- ✅ **ティッカーコード変更**: リアルタイムでの銘柄変更
- ✅ **移動平均線追加**: 複数の移動平均線を同時表示
- ✅ **カスタム移動平均**: 任意の期間の移動平均を追加
- ✅ **テクニカル指標**: RSI、ボリンジャーバンドのオン/オフ
- ✅ **期間設定**: 分析期間の動的変更

## 📋 セットアップ

### 1. Snowflakeマーケットプレイスデータの準備
1. Snowflakeアカウントにログイン
2. マーケットプレイスから金融データセットを取得
   - 推奨: **Cybersyn Financial & Economic Essentials**
   - または他の株式データプロバイダー
3. データセットをアカウントにインストール

**重要**: マーケットプレイスデータが利用できない場合、アプリケーションは自動的にサンプルデータを生成します。

### 2. データソースの設定
Cybersyn Financial & Economic Essentialsデータセットでは、価格データが`value`カラムに格納され、`variable_name`で種類を区別します：

```python
# 実際のCybersyn Financial & Economic Essentialsスキーマ
WITH stock_data AS (
    SELECT 
        date,
        ticker,
        variable_name,
        value
    FROM FINANCIAL__ECONOMIC_ESSENTIALS.CYBERSYN.STOCK_PRICE_TIMESERIES
    WHERE ticker = 'AAPL'
    AND value IS NOT NULL
),

closing_prices AS (
    SELECT date, ticker, value AS closing_price
    FROM stock_data
    WHERE variable_name = 'Post-Market Close'
),

opening_prices AS (
    SELECT date, ticker, value AS opening_price
    FROM stock_data
    WHERE variable_name = 'Pre-Market Open'
)
-- ... 他の価格データも同様に取得
```

**重要な variable_name の値:**
- `Post-Market Close`: 終値
- `Pre-Market Open`: 始値
- `High`: 最高値
- `Low`: 最安値
- `Nasdaq Volume`: 出来高（重要：`Volume`ではなく`Nasdaq Volume`）

**PIVOTクエリの使用例:**
```sql
-- GitHubハンズオンと同じPIVOT操作
SELECT 
    ticker,
    date,
    "Post-Market Close" AS close_price,
    "Pre-Market Open" AS open_price,
    "High" AS high_price,
    "Low" AS low_price,
    "Nasdaq Volume" AS volume
FROM (
    SELECT ticker, date, variable_name, value
    FROM FINANCE__ECONOMICS.CYBERSYN.STOCK_PRICE_TIMESERIES 
    WHERE ticker = 'AAPL'
    AND variable_name IN ('Nasdaq Volume', 'Post-Market Close', 'Pre-Market Open', 'High', 'Low')
)
PIVOT(
    MAX(value) 
    FOR variable_name IN (
        'Nasdaq Volume',
        'Post-Market Close',
        'Pre-Market Open',
        'High',
        'Low'
    )
)
ORDER BY date DESC;
```

## 🎮 使用方法

### 1. アプリケーションの起動
- SnowsightよりStreamlitを起動し、アップロードしたアプリを選択

### 2. 銘柄の選択
- **人気銘柄から選択**: AAPL, GOOGL, MSFT, AMZN, TSLA, NVDA, META, NFLX
- **ティッカーコード入力**: 任意の銘柄を直接入力

### 3. 分析期間の設定
- 1ヶ月、3ヶ月、6ヶ月、1年、2年から選択

### 4. 移動平均線の設定
- 5日移動平均（短期トレンド）
- 25日移動平均（中期トレンド）
- 50日移動平均（中長期トレンド）
- 200日移動平均（長期トレンド）
- カスタム移動平均（任意の期間）

### 5. テクニカル指標の表示
- **RSI**: 相対力指数（買われすぎ・売られすぎの判定）
- **ボリンジャーバンド**: 価格の変動幅を可視化
- **出来高**: 取引量の推移

## 📊 画面構成

### メインダッシュボード
- 現在価格、52週高値・安値、平均出来高のメトリクス
- インタラクティブな株価チャート
- 出来高グラフ
- RSIインジケーター

### サイドバー
- 銘柄選択
- 分析期間設定
- 移動平均線設定
- テクニカル指標の表示・非表示
- カスタム移動平均の追加

### データテーブル
- 日次データの詳細表示
- 統計情報の表示
- CSVダウンロード機能

## 🔧 カスタマイズ

### 新しい銘柄の追加
```python
popular_tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']
# 新しい銘柄を追加
```

### 追加のテクニカル指標
```python
def calculate_technical_indicators(df):
    # 新しい指標を追加
    # 例: MACD, ストキャスティクス等
    pass
```

### データソースの変更
実際のマーケットプレイスデータソースに合わせて、`get_stock_data`関数内のSQLクエリを変更してください。

## 🚨 注意事項

### データソースについて
- 実際のマーケットプレイスデータが利用できない場合、サンプルデータを生成します
- 本番環境では、実際のデータプロバイダーのスキーマに合わせて調整が必要です

### パフォーマンス
- 大量のデータを扱う場合は、適切なキャッシュ戦略を実装してください
- 長期間のデータ分析では、データの分割読み込みを検討してください

## 🔧 トラブルシューティング

### よくあるエラーと解決方法

**1. `invalid identifier 'CLOSE_PRICE'` エラー**
- **原因**: マーケットプレイスデータのカラム名が異なる
- **解決**: `get_stock_data()`関数内のSQLクエリのカラム名を実際のスキーマに合わせて調整

**2. `TypeError: max() got multiple values for argument '_emit_ast'`**
- **原因**: Snowparkの関数とPython組み込み関数の衝突
- **解決**: 修正済み（`builtins.max()`を使用）

**3. Snowflakeセッションの取得エラー**
- **原因**: Snowflakeの環境外で実行している
- **解決**: Snowflakeの環境内で実行するか、サンプルデータモードで使用

**4. 出来高データがすべて0で表示される**
- **原因**: `variable_name`が`'Volume'`ではなく`'Nasdaq Volume'`である
- **解決**: 修正済み（`'Nasdaq Volume'`を使用）
- **確認方法**: 
  ```sql
  -- 利用可能なvariable_nameを確認
  SELECT DISTINCT variable_name 
  FROM FINANCE__ECONOMICS.CYBERSYN.STOCK_PRICE_TIMESERIES 
  WHERE ticker = 'AAPL' 
  AND variable_name LIKE '%Volume%';
  ```

**5. データテーブルが表示されない**
- **原因**: カラム名の不一致やデータ型の問題
- **解決**: デバッグ情報を表示するよう修正済み
- **確認**: アプリ内のデバッグ情報でデータフレームの形状とカラム名を確認

### データソースの確認方法
```sql
-- マーケットプレイスで利用可能なデータベースを確認
SHOW DATABASES;

-- 特定のデータベースのスキーマを確認
SHOW SCHEMAS IN DATABASE FINANCIAL__ECONOMIC_ESSENTIALS;

-- テーブルの構造を確認
DESCRIBE TABLE FINANCIAL__ECONOMIC_ESSENTIALS.CYBERSYN.STOCK_PRICE_TIMESERIES;

-- 利用可能なvariable_nameを確認
SELECT DISTINCT variable_name 
FROM FINANCIAL__ECONOMIC_ESSENTIALS.CYBERSYN.STOCK_PRICE_TIMESERIES 
WHERE ticker = 'AAPL' 
LIMIT 20;

-- 特定の銘柄の価格データサンプルを確認
SELECT date, ticker, variable_name, value
FROM FINANCIAL__ECONOMIC_ESSENTIALS.CYBERSYN.STOCK_PRICE_TIMESERIES
WHERE ticker = 'AAPL'
AND variable_name IN ('Post-Market Close', 'Pre-Market Open', 'High', 'Low', 'Volume')
AND date >= CURRENT_DATE - 7
ORDER BY date DESC, variable_name;
```

## 💡 拡張アイデア

### 追加機能
- **アラート機能**: 価格や指標の閾値設定
- **複数銘柄比較**: 複数の銘柄を同時に分析
- **予測機能**: 機械学習を使った株価予測
- **レポート機能**: 分析結果のPDF出力
- **リアルタイム更新**: 定期的なデータ自動更新

### 高度な分析
- **ポートフォリオ分析**: 複数銘柄の組み合わせ分析
- **リスク分析**: VaR、シャープレシオ等の計算
- **バックテスト**: 投資戦略の検証
- **セクター分析**: 業界別の比較分析

## 📞 サポート

質問や問題がございましたら、以下までお気軽にお問い合わせください。

**本アプリケーションは、Snowflakeのマーケットプレイスデータを活用した分析体験のデモンストレーションとして作成されています。** 
