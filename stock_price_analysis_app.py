import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import builtins
from datetime import datetime, timedelta
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.functions import col, avg, count, sum
from snowflake.snowpark.functions import max as sf_max, min as sf_min

# AI_COMPLETE関数用のLLMモデル選択肢
AI_COMPLETE_MODELS = [
    "claude-4-sonnet",
    "llama4-maverick",
    "mistral-large2"
]

# Streamlitページ設定
st.set_page_config(
    page_title="📈米国株式分析アプリ",
    page_icon="📈",
    layout="wide"
)

# サイドバーでページ選択
st.sidebar.title("📊 分析メニュー")
page = st.sidebar.selectbox(
    "分析タイプを選択してください",
    ["単一銘柄分析", "複数銘柄比較"]
)

# アプリケーションのタイトル
st.title("📈米国株式分析アプリ")
st.markdown("---")

# Snowflakeセッションの取得
@st.cache_resource(ttl=600)
def get_snowflake_session():
    try:
        session = get_active_session()
        return session
    except Exception as e:
        st.error(f"Snowflakeセッションの取得に失敗しました: {str(e)}")
        return None

# 株式データの取得（マーケットプレイスデータを想定）
@st.cache_data(ttl=600)
def get_stock_data(ticker_symbol, days=365):
    """
    Snowflakeのマーケットプレイスから株式データを取得
    実際の実装では、利用可能なマーケットプレイスデータソースに合わせて調整してください
    """
    session = get_snowflake_session()
    if session is None:
        return generate_sample_data(ticker_symbol, days)
    
    # まず利用可能なvariable_nameを確認
    debug_query = f"""
    SELECT DISTINCT variable_name, COUNT(*) as count
    FROM SNOWFLAKE_PUBLIC_DATA_FREE.PUBLIC_DATA_FREE.STOCK_PRICE_TIMESERIES 
    WHERE ticker = '{ticker_symbol}'
    AND date >= CURRENT_DATE - 30
    GROUP BY variable_name
    ORDER BY count DESC
    """
    
    try:
        debug_result = session.sql(debug_query).collect()
        available_variables = [row['VARIABLE_NAME'] for row in debug_result]
        # st.write(f"🔍 デバッグ: {ticker_symbol} で利用可能な variable_name:", available_variables)
    except Exception as e:
        # st.write(f"🔍 デバッグクエリエラー: {str(e)}")
        available_variables = []

    # Cybersyn Financial & Economic Essentials の実際のスキーマに基づくクエリ
    query_patterns = [
        # パターン1: JOIN方式（Snowpark対応、確実に動作）
        f"""
        WITH stock_data AS (
            SELECT 
                date,
                ticker,
                variable_name,
                value
            FROM SNOWFLAKE_PUBLIC_DATA_FREE.PUBLIC_DATA_FREE.STOCK_PRICE_TIMESERIES
            WHERE ticker = '{ticker_symbol}'
            AND date >= CURRENT_DATE - {days}
            AND value IS NOT NULL
        ),
        
        opening_prices AS (
            SELECT date, ticker, value AS open_price
            FROM stock_data
            WHERE variable_name = 'Pre-Market Open'
        ),
        
        closing_prices AS (
            SELECT date, ticker, value AS close_price
            FROM stock_data
            WHERE variable_name = 'Post-Market Close'
        ),
        
        high_prices AS (
            SELECT date, ticker, value AS high_price
            FROM stock_data
            WHERE variable_name = 'High'
        ),
        
        low_prices AS (
            SELECT date, ticker, value AS low_price
            FROM stock_data
            WHERE variable_name = 'Low'
        ),
        
        volumes AS (
            SELECT date, ticker, value AS volume
            FROM stock_data
            WHERE variable_name = 'Nasdaq Volume'
        ),
        
        date_ticker_combinations AS (
            SELECT DISTINCT date, ticker
            FROM stock_data
        )
        
        SELECT 
            dtc.date,
            dtc.ticker,
            COALESCE(cp.close_price, 0) AS close_price,
            COALESCE(op.open_price, cp.close_price, 0) AS open_price,
            COALESCE(hp.high_price, cp.close_price, 0) AS high_price,
            COALESCE(lp.low_price, cp.close_price, 0) AS low_price,
            COALESCE(v.volume, 0) AS volume,
            COALESCE(cp.close_price, 0) AS adjusted_close
        FROM date_ticker_combinations dtc
        LEFT JOIN opening_prices op ON dtc.date = op.date AND dtc.ticker = op.ticker
        LEFT JOIN closing_prices cp ON dtc.date = cp.date AND dtc.ticker = cp.ticker
        LEFT JOIN high_prices hp ON dtc.date = hp.date AND dtc.ticker = hp.ticker
        LEFT JOIN low_prices lp ON dtc.date = lp.date AND dtc.ticker = lp.ticker
        LEFT JOIN volumes v ON dtc.date = v.date AND dtc.ticker = v.ticker
        WHERE cp.close_price IS NOT NULL
        ORDER BY dtc.date DESC
        LIMIT 1000
        """,
        
        # パターン2: JOIN方式（Nasdaq Volume使用）
        f"""
        WITH stock_data AS (
            SELECT 
                date,
                ticker,
                variable_name,
                value
            FROM SNOWFLAKE_PUBLIC_DATA_FREE.PUBLIC_DATA_FREE.STOCK_PRICE_TIMESERIES
            WHERE ticker = '{ticker_symbol}'
            AND date >= CURRENT_DATE - {days}
            AND value IS NOT NULL
        ),
        
        -- 各価格データを取得
        opening_prices AS (
            SELECT date, ticker, value AS opening_price
            FROM stock_data
            WHERE variable_name = 'Pre-Market Open'
        ),
        
        closing_prices AS (
            SELECT date, ticker, value AS closing_price
            FROM stock_data
            WHERE variable_name = 'Post-Market Close'
        ),
        
        high_prices AS (
            SELECT date, ticker, value AS high_price
            FROM stock_data
            WHERE variable_name = 'High'
        ),
        
        low_prices AS (
            SELECT date, ticker, value AS low_price
            FROM stock_data
            WHERE variable_name = 'Low'
        ),
        
        volume_data AS (
            SELECT date, ticker, value AS volume
            FROM stock_data
            WHERE variable_name = 'Nasdaq Volume'
        )
        
        -- LEFT JOINで各価格データを結合
        SELECT 
            c.date,
            c.ticker,
            c.closing_price AS close_price,
            COALESCE(o.opening_price, c.closing_price) AS open_price,
            COALESCE(h.high_price, c.closing_price) AS high_price,
            COALESCE(l.low_price, c.closing_price) AS low_price,
            COALESCE(v.volume, 0) AS volume,
            c.closing_price AS adjusted_close
        FROM closing_prices c
        LEFT JOIN opening_prices o ON c.date = o.date AND c.ticker = o.ticker
        LEFT JOIN high_prices h ON c.date = h.date AND c.ticker = h.ticker
        LEFT JOIN low_prices l ON c.date = l.date AND c.ticker = l.ticker
        LEFT JOIN volume_data v ON c.date = v.date AND c.ticker = v.ticker
        ORDER BY c.date DESC
        LIMIT 1000
        """,
        
        # パターン3: 異なるvolume名でのフォールバック
        f"""
        WITH stock_data AS (
            SELECT 
                date,
                ticker,
                variable_name,
                value
            FROM SNOWFLAKE_PUBLIC_DATA_FREE.PUBLIC_DATA_FREE.STOCK_PRICE_TIMESERIES
            WHERE ticker = '{ticker_symbol}'
            AND date >= CURRENT_DATE - {days}
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
        ),
        
        high_prices AS (
            SELECT date, ticker, value AS high_price
            FROM stock_data
            WHERE variable_name = 'High'
        ),
        
        low_prices AS (
            SELECT date, ticker, value AS low_price
            FROM stock_data
            WHERE variable_name = 'Low'
        ),
        
        volume_data AS (
            SELECT date, ticker, value AS volume
            FROM stock_data
            WHERE variable_name IN ('Volume', 'Total Volume', 'Exchange Volume')
        )
        
        SELECT 
            c.date,
            c.ticker,
            c.closing_price AS close_price,
            COALESCE(o.opening_price, c.closing_price) AS open_price,
            COALESCE(h.high_price, c.closing_price) AS high_price,
            COALESCE(l.low_price, c.closing_price) AS low_price,
            COALESCE(v.volume, 0) AS volume,
            c.closing_price AS adjusted_close
        FROM closing_prices c
        LEFT JOIN opening_prices o ON c.date = o.date AND c.ticker = o.ticker
        LEFT JOIN high_prices h ON c.date = h.date AND c.ticker = h.ticker
        LEFT JOIN low_prices l ON c.date = l.date AND c.ticker = l.ticker
        LEFT JOIN volume_data v ON c.date = v.date AND c.ticker = v.ticker
        ORDER BY c.date DESC
        LIMIT 1000
        """,
        
        # パターン4: 単一のPRICE/VOLUMEカラム構造
        f"""
        WITH price_volume_data AS (
            SELECT 
                date,
                ticker,
                MAX(CASE WHEN variable_name = 'Post-Market Close' THEN value END) AS price,
                MAX(CASE WHEN variable_name = 'Nasdaq Volume' THEN value END) AS volume
            FROM SNOWFLAKE_PUBLIC_DATA_FREE.PUBLIC_DATA_FREE.STOCK_PRICE_TIMESERIES
            WHERE ticker = '{ticker_symbol}'
            AND date >= CURRENT_DATE - {days}
            AND value IS NOT NULL
            GROUP BY date, ticker
        )
        SELECT 
            date,
            ticker,
            price AS close_price,
            price AS open_price,
            price AS high_price,
            price AS low_price,
            COALESCE(volume, 0) AS volume,
            price AS adjusted_close
        FROM price_volume_data
        WHERE price IS NOT NULL
        ORDER BY date DESC
        LIMIT 1000
        """
    ]
    
    for i, query in enumerate(query_patterns):
        try:
            # データを取得
            df = session.sql(query).to_pandas()
            
            if not df.empty:
                # カラム名を小文字に統一
                df.columns = df.columns.str.lower()
                
                # 日付列の型変換
                df['date'] = pd.to_datetime(df['date'])
                
                # TICKERカラムの処理
                if 'ticker' not in df.columns:
                    df['ticker'] = ticker_symbol
                
                # 数値カラムの型変換
                numeric_columns = ['close_price', 'open_price', 'high_price', 'low_price', 'volume', 'adjusted_close']
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # 必要な列を選択
                required_columns = ['date', 'ticker', 'close_price', 'open_price', 'high_price', 'low_price', 'volume', 'adjusted_close']
                df = df[required_columns]
                
                # データの前処理
                df = df.dropna(subset=['close_price'])  # 終値がNullの行を削除
                df = df.sort_values('date')  # 日付順にソート
                
                # デバッグ情報を出力
                # st.write(f"**パターン {i+1} デバッグ情報**:")
                # st.write(f"- 取得行数: {len(df)}")
                # st.write(f"- Volume列の型: {df['volume'].dtype}")
                # st.write(f"- Volume列の統計: {df['volume'].describe()}")
                # st.write(f"- Volume列の最初の5つの値: {df['volume'].head().tolist()}")
                
                # volume列に0でない値があるかチェック
                # non_zero_volume = df[df['volume'] > 0]['volume'].count()
                # st.write(f"- 0でない出来高データの数: {non_zero_volume}")
                
                if df['volume'].sum() == 0:
                    st.warning(f"⚠️ パターン {i+1}: 出来高データがすべて0または欠損値です")
                
                # st.success(f"データソースパターン {i+1} でデータを取得しました ({len(df)} 件)")
                df_with_indicators = calculate_technical_indicators(df)
                
                # st.write(f"**パターン {i+1} デバッグ情報**:")
                # st.write(f"- 取得行数: {len(df)}")
                # st.write(f"- Volume列の型: {df['volume'].dtype}")
                # st.write(f"- Volume列の統計: {df['volume'].describe()}")
                # st.write(f"- Volume列の最初の5つの値: {df['volume'].head().tolist()}")
                # st.write(f"- 0でない出来高データの数: {(df['volume'] > 0).sum()}")
                
                return df_with_indicators
                
        except Exception as e:
            st.warning(f"データソースパターン {i+1} でエラー: {str(e)}")
            continue
    
    # すべてのパターンで失敗した場合はサンプルデータを生成
    st.warning("マーケットプレイスデータの取得に失敗しました。サンプルデータを使用します。")
    st.info("""
    **実際のマーケットプレイスデータを使用するには:**
    1. Snowflakeマーケットプレイスで金融データセットを取得
    2. 推奨: 'Cybersyn Financial & Economic Essentials' データセット
    3. 上記のSQLクエリを実際のデータソースに合わせて調整
    4. 必要に応じてデータベース名やスキーマ名を更新
    """)
    return generate_sample_data(ticker_symbol, days)

# サンプルデータの生成（デモ用）
def generate_sample_data(ticker_symbol, days=365):
    """
    デモ用のサンプル株式データを生成
    """
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), 
                         end=datetime.now(), freq='D')
    
    # 基本価格の設定
    base_prices = {
        'AAPL': 150, 'GOOGL': 2500, 'MSFT': 300, 'AMZN': 3000,
        'TSLA': 200, 'NVDA': 400, 'META': 250, 'NFLX': 400
    }
    
    base_price = base_prices.get(ticker_symbol, 100)
    
    # ランダムウォークで価格を生成
    np.random.seed(42)  # 再現性のため
    price_changes = np.random.normal(0, 0.02, len(dates))
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(builtins.max(new_price, 1))  # 最低価格を1に設定
    
    # データフレームの作成
    df = pd.DataFrame({
        'date': dates,
        'ticker': ticker_symbol,
        'close_price': prices,
        'open_price': [p * (1 + np.random.normal(0, 0.001)) for p in prices],
        'high_price': [p * (1 + abs(np.random.normal(0, 0.015))) for p in prices],
        'low_price': [p * (1 - abs(np.random.normal(0, 0.015))) for p in prices],
        'volume': np.random.randint(1000000, 10000000, len(dates)),
        'adjusted_close': prices
    })
    
    return df

# 移動平均の計算
def calculate_moving_averages(df, periods):
    """
    指定された期間の移動平均を計算
    """
    for period in periods:
        df[f'ma_{period}'] = df['close_price'].rolling(window=period).mean()
    return df

# テクニカル指標の計算
def calculate_technical_indicators(df):
    """
    基本的なテクニカル指標を計算
    """
    # RSI
    delta = df['close_price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # ボリンジャーバンド
    df['bb_middle'] = df['close_price'].rolling(window=20).mean()
    bb_std = df['close_price'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    
    return df

# 単一銘柄分析の関数を追加
def single_stock_analysis():
    """単一銘柄分析を実行する関数"""
    # サイドバーでの設定
    st.sidebar.header("📊 分析設定")
    
    # 人気銘柄の事前定義リスト
    popular_stocks = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "NFLX", "UBER", "SHOP"]
    
    # 銘柄選択
    selected_stock = st.sidebar.selectbox("銘柄を選択", popular_stocks)
    
    # 手動入力オプション
    manual_input = st.sidebar.text_input("または手動で銘柄コードを入力", placeholder="例: AAPL")
    ticker_symbol = manual_input.upper() if manual_input else selected_stock
    
    # 期間選択
    period_options = {
        "1ヶ月": 30,
        "3ヶ月": 90,
        "6ヶ月": 180,
        "1年": 365,
        "2年": 730
    }
    
    selected_period = st.sidebar.selectbox("期間を選択", list(period_options.keys()), index=2)
    days = period_options[selected_period]
    
    # 移動平均の設定
    st.sidebar.subheader("📊 移動平均設定")
    ma_periods = []
    
    if st.sidebar.checkbox("5日移動平均", value=True):
        ma_periods.append(5)
    if st.sidebar.checkbox("25日移動平均", value=True):
        ma_periods.append(25)
    if st.sidebar.checkbox("50日移動平均", value=True):
        ma_periods.append(50)
    if st.sidebar.checkbox("200日移動平均", value=False):
        ma_periods.append(200)
    
    # カスタム移動平均
    custom_ma = st.sidebar.number_input("カスタム移動平均（日）", min_value=2, max_value=500, value=None, step=1)
    if custom_ma:
        ma_periods.append(custom_ma)
    
    # RSI設定
    st.sidebar.subheader("📊 RSI設定")
    show_rsi = st.sidebar.checkbox("RSIを表示", value=True)
    rsi_period = st.sidebar.number_input("RSI期間（日）", min_value=2, max_value=100, value=14, step=1)
    
    # AI分析設定
    st.sidebar.subheader("🤖 AI分析設定")
    enable_ai_analysis = st.sidebar.checkbox("AI分析を表示", value=True)
    if enable_ai_analysis:
        selected_model = st.sidebar.selectbox(
            "AIモデルを選択", 
            AI_COMPLETE_MODELS, 
            index=0,
            help="分析に使用するAIモデルを選択してください"
        )
        st.sidebar.info("🤖 SnowflakeのAI_COMPLETEを使用して株価分析と予測を生成します")
    
    # データ取得
    st.info(f"📊 **{ticker_symbol}** の **{selected_period}** 間のデータを取得中...")
    
    # データの取得
    with st.spinner("データを取得中..."):
        df = get_stock_data(ticker_symbol, days)
    
    if df.empty:
        st.error("データの取得に失敗しました。")
        return
    
    # 移動平均の計算
    df = calculate_moving_averages(df, ma_periods)
    
    # テクニカル指標の計算
    df = calculate_technical_indicators(df)
    
    # 残りの分析処理を続行
    display_single_stock_analysis(df, ticker_symbol, show_rsi, ma_periods, enable_ai_analysis, selected_model if enable_ai_analysis else None)

def display_single_stock_analysis(df, ticker_symbol, show_rsi, ma_periods, enable_ai_analysis, selected_model):
    """単一銘柄分析の表示処理"""
    # メイン表示エリア
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_price = df['close_price'].iloc[-1]
        if len(df) > 1:
            prev_price = df['close_price'].iloc[-2]
            price_change = ((current_price - prev_price) / prev_price * 100)
            delta_str = f"{price_change:.2f}%"
        else:
            delta_str = "N/A"
        
        st.metric(
            label="現在価格",
            value=f"${current_price:.2f}",
            delta=delta_str
        )
    
    with col2:
        if len(df) > 1:
            high_price = df['high_price'].iloc[-1]
            st.metric(
                label="高値",
                value=f"${high_price:.2f}"
            )
        else:
            st.metric(
                label="高値",
                value="N/A"
            )
    
    with col3:
        if len(df) > 1:
            low_price = df['low_price'].iloc[-1]
            st.metric(
                label="安値",
                value=f"${low_price:.2f}"
            )
        else:
            st.metric(
                label="安値",
                value="N/A"
            )
    
    with col4:
        if len(df) > 1:
            volume = df['volume'].iloc[-1]
            st.metric(
                label="出来高",
                value=f"{volume:,.0f}" if volume > 0 else "N/A"
            )
        else:
            st.metric(
                label="出来高",
                value="N/A"
            )
    
    # AI分析の表示（条件付き）
    if enable_ai_analysis:
        st.subheader("🤖 AI株価分析・予測")
        
        # AI分析を実行
        with st.spinner("🤖 AIが株価を分析中..."):
            ai_analysis = generate_ai_stock_analysis(df, ticker_symbol, selected_model)
        
        # AI分析結果を表示
        # 背景色を薄いブルーに設定
        st.markdown(f"""
        <div style="background-color: #E6F3FF; padding: 15px; border-radius: 10px; border-left: 4px solid #0066CC;">
        <h4 style="color: #0066CC; margin-bottom: 10px;">🤖 AI分析結果:</h4>
        <div style="color: #333333; line-height: 1.6;">
        {ai_analysis}
        </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
    
    # 価格チャート
    st.subheader("📈 価格チャート")
    
    # チャート作成
    rows = 2 if show_rsi else 1
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("価格", "RSI") if show_rsi else ("価格",),
        row_heights=[0.7, 0.3] if show_rsi else [1.0]
    )
    
    # 価格データ
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['close_price'],
            mode='lines',
            name='終値',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # 移動平均を追加
    colors = ['red', 'green', 'orange', 'purple', 'brown']
    for i, ma_period in enumerate(ma_periods):
        col_name = f'ma_{ma_period}'
        if col_name in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df[col_name],
                    mode='lines',
                    name=f'{ma_period}日移動平均',
                    line=dict(color=colors[i % len(colors)], width=1, dash='dash')
                ),
                row=1, col=1
            )
    
    # ボリンジャーバンドを追加
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['bb_upper'],
                mode='lines',
                name='ボリンジャーバンド上限',
                line=dict(color='gray', width=1),
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['bb_lower'],
                mode='lines',
                name='ボリンジャーバンド下限',
                line=dict(color='gray', width=1),
                fill='tonexty',
                fillcolor='rgba(128,128,128,0.1)',
                showlegend=True
            ),
            row=1, col=1
        )
    
    # RSIを追加
    if show_rsi and 'rsi' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['rsi'],
                mode='lines',
                name='RSI',
                line=dict(color='purple', width=2)
            ),
            row=2, col=1
        )
        
        # RSIのオーバーボート/オーバーソールドライン
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # レイアウト更新
    fig.update_layout(
        title=f"{ticker_symbol} - 株価チャート",
        xaxis_title="日付",
        yaxis_title="価格 (USD)",
        height=800 if show_rsi else 600,
        showlegend=True
    )
    
    # Y軸の範囲設定
    if show_rsi:
        fig.update_yaxes(range=[0, 100], row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 出来高チャート
    st.subheader("📊 出来高")
    
    # 出来高データの詳細確認
    if 'volume' in df.columns:
        if df['volume'].sum() > 0:
            # 出来高チャート用のサブプロットを作成
            fig_volume = make_subplots(rows=1, cols=1)
            
            fig_volume.add_trace(
                go.Bar(
                    x=df['date'],
                    y=df['volume'],
                    name='出来高',
                    marker_color='rgba(55, 128, 191, 0.7)',
                    showlegend=False
                )
            )
            
            fig_volume.update_layout(
                title=f"{ticker_symbol} - 出来高",
                xaxis_title="日付",
                yaxis_title="出来高",
                height=300
            )
            
            st.plotly_chart(fig_volume, use_container_width=True)
        else:
            st.warning("出来高データがありません")
    else:
        st.warning("出来高データが利用できません")
    
    # データテーブルの表示
    display_data_table(df, ticker_symbol)

def display_data_table(df, ticker_symbol):
    """データテーブルの表示"""
    st.subheader("📊 データテーブル")
    
    # データの先頭を表示
    if not df.empty:
        st.write("**先頭5行のデータ**:")
        st.dataframe(df.head(), use_container_width=True)
    
    # 表示する列を選択
    available_columns = ['date', 'ticker', 'close_price', 'open_price', 'high_price', 'low_price', 'volume', 'adjusted_close']
    technical_columns = [col for col in df.columns if col.startswith('ma_') or col in ['rsi', 'bb_upper', 'bb_lower', 'bb_middle']]
    all_display_columns = available_columns + technical_columns
    
    # 実際に存在する列のみを選択
    existing_columns = [col for col in all_display_columns if col in df.columns]
    
    if existing_columns:
        selected_columns = st.multiselect(
            "表示する列を選択してください",
            existing_columns,
            default=existing_columns[:8]  # 最初の8列をデフォルトで選択
        )
        
        if selected_columns:
            display_df = df[selected_columns].copy()
            
            # 日付の形式を調整
            if 'date' in display_df.columns:
                display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
            
            # 数値の丸め
            numeric_columns = display_df.select_dtypes(include=[np.number]).columns
            display_df[numeric_columns] = display_df[numeric_columns].round(2)
            
            st.write("**詳細データテーブル**:")
            st.dataframe(display_df.head(20), use_container_width=True)
            
            # 出来高データの詳細確認
            if 'volume' in display_df.columns:
                if display_df['volume'].sum() == 0:
                    st.warning("⚠️ 出来高データがすべて0です。データソースでvolume列の値を確認してください。")
            
            # CSVダウンロード
            csv_data = display_df.to_csv(index=False)
            st.download_button(
                label="📥 CSVダウンロード",
                data=csv_data,
                file_name=f"{ticker_symbol}_stock_data.csv",
                mime="text/csv"
            )
    else:
        st.warning("表示可能な列がありません。データを確認してください。")
    
    # サマリー統計
    st.subheader("📊 サマリー統計")
    
    if not df.empty:
        summary_stats = df[['close_price', 'high_price', 'low_price', 'volume']].describe()
        st.dataframe(summary_stats, use_container_width=True)

# AI分析機能の追加
def generate_ai_stock_analysis(df, ticker_symbol, selected_model):
    """
    SnowflakeのAI_COMPLETEを使用して株価分析と予測を生成
    """
    session = get_snowflake_session()
    if session is None:
        return "AI分析機能はSnowflakeセッションが必要です。"
    
    try:
        # 最新のデータを取得
        if len(df) < 5:
            return "分析に十分なデータがありません。"
        
        # 技術指標の現在値を取得
        current_price = df['close_price'].iloc[-1]
        prev_price = df['close_price'].iloc[-2] if len(df) > 1 else current_price
        price_change = ((current_price - prev_price) / prev_price * 100) if prev_price > 0 else 0
        
        # 移動平均情報
        ma_5 = df['ma_5'].iloc[-1] if 'ma_5' in df.columns and not pd.isna(df['ma_5'].iloc[-1]) else current_price
        ma_25 = df['ma_25'].iloc[-1] if 'ma_25' in df.columns and not pd.isna(df['ma_25'].iloc[-1]) else current_price
        
        # RSI情報
        rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns and not pd.isna(df['rsi'].iloc[-1]) else 50
        
        # 出来高情報
        volume = df['volume'].iloc[-1] if 'volume' in df.columns else 0
        avg_volume = df['volume'].mean() if 'volume' in df.columns else 0
        
        # AI分析のためのプロンプト作成（SQLインジェクション対策）
        analysis_prompt = f"""以下の株価データを分析してください：

銘柄: {ticker_symbol}
現在価格: ${current_price:.2f}
前日比: {price_change:.2f}%
5日移動平均: ${ma_5:.2f}
25日移動平均: ${ma_25:.2f}
RSI: {rsi:.1f}
出来高: {volume:,.0f}
平均出来高: {avg_volume:,.0f}

テクニカル分析の観点から、今後の株価動向と投資判断を3-4文で要約してください。
RSIが70以上なら過熱感、30以下なら過売り感を考慮してください。
移動平均との関係も分析に含めてください。"""
        
        # プロンプトのエスケープ処理
        escaped_prompt = analysis_prompt.replace("'", "''")
        
        # AI_COMPLETEを使用して分析を実行（選択されたモデルを使用）
        try:
            ai_query = f"""
            SELECT AI_COMPLETE(
                '{selected_model}',
                '{escaped_prompt}'
            ) AS analysis_result
            """
            
            result = session.sql(ai_query).collect()
            
            if result and len(result) > 0 and result[0]['ANALYSIS_RESULT']:
                analysis = result[0]['ANALYSIS_RESULT']
                if analysis and analysis.strip():
                    # 前後の""を削除し、改行を修正
                    cleaned_analysis = analysis.strip()
                    if cleaned_analysis.startswith('"') and cleaned_analysis.endswith('"'):
                        cleaned_analysis = cleaned_analysis[1:-1]
                    cleaned_analysis = cleaned_analysis.replace('\\n\\n', '<br><br>').replace('\\n', '<br>')
                    return f"<strong style='color: #0066CC; font-size: 1.1em;'>モデル: {selected_model}</strong><br><br>{cleaned_analysis}"
                    
        except Exception as model_error:
            print(f"モデル {selected_model} でエラー: {str(model_error)}")
            # フォールバック：従来のモデルを試行
            fallback_models = ['snowflake-arctic', 'llama2-70b-chat', 'mixtral-8x7b']
            for model in fallback_models:
                try:
                    ai_query = f"""
                    SELECT AI_COMPLETE(
                        '{model}',
                        '{escaped_prompt}'
                    ) AS analysis_result
                    """
                    
                    result = session.sql(ai_query).collect()
                    
                    if result and len(result) > 0 and result[0]['ANALYSIS_RESULT']:
                        analysis = result[0]['ANALYSIS_RESULT']
                        if analysis and analysis.strip():
                            # 前後の""を削除し、改行を修正
                            cleaned_analysis = analysis.strip()
                            if cleaned_analysis.startswith('"') and cleaned_analysis.endswith('"'):
                                cleaned_analysis = cleaned_analysis[1:-1]
                            cleaned_analysis = cleaned_analysis.replace('\\n\\n', '<br><br>').replace('\\n', '<br>')
                            return f"<strong style='color: #0066CC; font-size: 1.1em;'>モデル: {model} (フォールバック)</strong><br><br>{cleaned_analysis}"
                            
                except Exception as fallback_error:
                    print(f"フォールバックモデル {model} でエラー: {str(fallback_error)}")
                    continue
        
        return "選択されたAIモデルで分析を生成できませんでした。"
            
    except Exception as e:
        return f"AI分析エラー: {str(e)}"

def generate_ai_comparison_analysis(comparison_data, normalize_prices, selected_model):
    """
    複数銘柄比較のAI分析を生成
    """
    session = get_snowflake_session()
    if session is None:
        return "AI分析機能はSnowflakeセッションが必要です。"
    
    try:
        # 各銘柄のパフォーマンスを計算
        performance_summary = []
        
        for ticker, data in comparison_data.items():
            if not data.empty:
                df = data.copy()
                df = df.sort_values('date')
                
                first_price = df['close_price'].iloc[0]
                last_price = df['close_price'].iloc[-1]
                change_percent = ((last_price - first_price) / first_price * 100) if first_price > 0 else 0
                
                performance_summary.append(f"{ticker}: {change_percent:.2f}%")
        
        # AI分析のためのプロンプト作成（SQLインジェクション対策）
        analysis_prompt = f"""以下の複数銘柄のパフォーマンスを分析して、投資判断を日本語で簡潔に箇条書きで要約してください。

パフォーマンス:
{', '.join(performance_summary)}

標準化表示: {'はい' if normalize_prices else 'いいえ'}

以下の形式で回答してください：
• 最高パフォーマンス: [銘柄名] ([パフォーマンス]%)
• 最低パフォーマンス: [銘柄名] ([パフォーマンス]%)
• 投資戦略: [1-2文で簡潔に]
• 今後の見通し: [1-2文で簡潔に]
• 推奨度: [高/中/低] - [理由を1文で]"""
        
        # プロンプトのエスケープ処理
        escaped_prompt = analysis_prompt.replace("'", "''")
        
        # AI_COMPLETEを使用して分析を実行（選択されたモデルを使用）
        try:
            ai_query = f"""
            SELECT AI_COMPLETE(
                '{selected_model}',
                '{escaped_prompt}'
            ) AS analysis_result
            """
            
            result = session.sql(ai_query).collect()
            
            if result and len(result) > 0 and result[0]['ANALYSIS_RESULT']:
                analysis = result[0]['ANALYSIS_RESULT']
                if analysis and analysis.strip():
                    # 前後の""を削除し、改行を修正
                    cleaned_analysis = analysis.strip()
                    if cleaned_analysis.startswith('"') and cleaned_analysis.endswith('"'):
                        cleaned_analysis = cleaned_analysis[1:-1]
                    cleaned_analysis = cleaned_analysis.replace('\\n', '<br>')
                    return f"<strong style='color: #0066CC; font-size: 1.1em;'>モデル: {selected_model}</strong><br><br>{cleaned_analysis}"
                    
        except Exception as model_error:
            print(f"モデル {selected_model} でエラー: {str(model_error)}")
            # フォールバック：従来のモデルを試行
            fallback_models = ['snowflake-arctic', 'llama2-70b-chat', 'mixtral-8x7b']
            for model in fallback_models:
                try:
                    ai_query = f"""
                    SELECT AI_COMPLETE(
                        '{model}',
                        '{escaped_prompt}'
                    ) AS analysis_result
                    """
                    
                    result = session.sql(ai_query).collect()
                    
                    if result and len(result) > 0 and result[0]['ANALYSIS_RESULT']:
                        analysis = result[0]['ANALYSIS_RESULT']
                        if analysis and analysis.strip():
                            # 前後の""を削除し、改行を修正
                            cleaned_analysis = analysis.strip()
                            if cleaned_analysis.startswith('"') and cleaned_analysis.endswith('"'):
                                cleaned_analysis = cleaned_analysis[1:-1]
                            cleaned_analysis = cleaned_analysis.replace('\\n', '<br>')
                            return f"<strong style='color: #0066CC; font-size: 1.1em;'>モデル: {model} (フォールバック)</strong><br><br>{cleaned_analysis}"
                            
                except Exception as fallback_error:
                    print(f"フォールバックモデル {model} でエラー: {str(fallback_error)}")
                    continue
        
        return "選択されたAIモデルで分析を生成できませんでした。"
            
    except Exception as e:
        return f"AI分析エラー: {str(e)}"

# 複数銘柄比較の関数定義
def create_comparison_chart(comparison_data, normalize_prices):
    """複数銘柄の比較チャートを作成"""
    st.subheader("📈 価格比較チャート")
    
    fig = go.Figure()
    
    for ticker, data in comparison_data.items():
        if not data.empty:
            df = data.copy()
            df = df.sort_values('date')
            
            if normalize_prices:
                # 標準化：最初の価格を100として、その後の変化率を表示
                first_price = df['close_price'].iloc[0]
                if first_price > 0:
                    df['normalized_price'] = (df['close_price'] / first_price) * 100
                    y_values = df['normalized_price']
                    y_title = "標準化価格 (スタート=100)"
                else:
                    y_values = df['close_price']
                    y_title = "価格 (USD)"
            else:
                y_values = df['close_price']
                y_title = "価格 (USD)"
            
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=y_values,
                    mode='lines',
                    name=ticker,
                    line=dict(width=2),
                    hovertemplate=f"<b>{ticker}</b><br>" +
                                f"日付: %{{x}}<br>" +
                                f"価格: %{{y:.2f}}<br>" +
                                "<extra></extra>"
                )
            )
    
    # チャートタイトルを標準化の状態に応じて設定
    if normalize_prices:
        chart_title = "複数銘柄価格比較（標準化表示）"
        chart_subtitle = "※ 各銘柄の開始価格を100として相対的な変化率を表示"
    else:
        chart_title = "複数銘柄価格比較（実際の株価）"
        chart_subtitle = "※ 実際の株価を表示"
    
    fig.update_layout(
        title=chart_title,
        xaxis_title="日付",
        yaxis_title=y_title,
        hovermode='x unified',
        showlegend=True,
        height=600
    )
    
    # チャートの説明を表示
    st.info(chart_subtitle)
    
    st.plotly_chart(fig, use_container_width=True)

def create_comparison_table(comparison_data, normalize_prices):
    """複数銘柄の比較テーブルを作成"""
    if normalize_prices:
        st.subheader("📊 パフォーマンス比較テーブル（標準化表示）")
        st.info("📊 各銘柄の開始価格を100として相対的な変化率を表示")
    else:
        st.subheader("📊 パフォーマンス比較テーブル（実際の株価）")
        st.info("💰 実際の株価を表示")
    
    summary_data = []
    
    for ticker, data in comparison_data.items():
        if not data.empty:
            df = data.copy()
            df = df.sort_values('date')
            
            first_price = df['close_price'].iloc[0]
            last_price = df['close_price'].iloc[-1]
            
            if first_price > 0:
                change_percent = ((last_price - first_price) / first_price) * 100
                
                if normalize_prices:
                    # 標準化表示の場合
                    summary_data.append({
                        "銘柄": ticker,
                        "開始価格": "100.00",
                        "最終価格": f"{((last_price / first_price) * 100):.2f}",
                        "変化率": f"{change_percent:.2f}%",
                        "最高値": f"{((df['high_price'].max() / first_price) * 100):.2f}",
                        "最安値": f"{((df['low_price'].min() / first_price) * 100):.2f}",
                        "データ期間": f"{len(df)} 日"
                    })
                else:
                    # 実際の株価表示の場合
                    summary_data.append({
                        "銘柄": ticker,
                        "開始価格": f"${first_price:.2f}",
                        "最終価格": f"${last_price:.2f}",
                        "変化率": f"{change_percent:.2f}%",
                        "最高値": f"${df['high_price'].max():.2f}",
                        "最安値": f"${df['low_price'].min():.2f}",
                        "データ期間": f"{len(df)} 日"
                    })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        # 変化率で降順ソート
        summary_df = summary_df.sort_values("変化率", key=lambda x: x.str.rstrip('%').astype(float), ascending=False)
        st.dataframe(summary_df, use_container_width=True)
        
        # 上位パフォーマンス銘柄を表示
        if len(summary_df) > 1:
            best_performer = summary_df.iloc[0]
            worst_performer = summary_df.iloc[-1]
            
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"🏆 **最高パフォーマンス**: {best_performer['銘柄']} ({best_performer['変化率']})")
            with col2:
                st.error(f"📉 **最低パフォーマンス**: {worst_performer['銘柄']} ({worst_performer['変化率']})")
    else:
        st.warning("⚠️ 比較用のデータがありません")

# 選択されたページに応じて表示を切り替え
if page == "単一銘柄分析":
    st.header("📊 単一銘柄分析")
    single_stock_analysis()

elif page == "複数銘柄比較":
    st.header("📈 複数銘柄比較分析")
    
    # サイドバーでの設定
    st.sidebar.header("📊 比較設定")
    
    # 期間選択
    period_options = {
        "1ヶ月": 30,
        "3ヶ月": 90,
        "6ヶ月": 180,
        "1年": 365,
        "2年": 730
    }
    
    selected_period = st.sidebar.selectbox("期間を選択", list(period_options.keys()), index=2)
    days = period_options[selected_period]
    
    # 表示オプション
    st.sidebar.subheader("📊 表示オプション")
    normalize_prices = st.sidebar.checkbox("価格を標準化する（スタート地点を100とする）", value=False)
    
    # 標準化の説明
    if normalize_prices:
        st.sidebar.info("📊 標準化ON：各銘柄の開始価格を100として相対的な変化率を表示")
    else:
        st.sidebar.info("💰 標準化OFF：実際の株価を表示")
    
    # AI分析設定
    st.sidebar.subheader("🤖 AI分析設定")
    enable_ai_analysis = st.sidebar.checkbox("AI分析を表示", value=True)
    if enable_ai_analysis:
        selected_model = st.sidebar.selectbox(
            "AIモデルを選択", 
            AI_COMPLETE_MODELS, 
            index=0,
            help="分析に使用するAIモデルを選択してください"
        )
        st.sidebar.info("🤖 SnowflakeのAI_COMPLETEを使用して複数銘柄の比較分析を生成します")
    
    # 複数銘柄選択
    st.subheader("🎯 銘柄選択")
    
    # 人気銘柄の事前定義リスト
    popular_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "UBER", "SHOP"]
    
    # 銘柄選択（複数選択可能）
    selected_tickers = st.multiselect(
        "比較する銘柄を選択してください（複数選択可能）",
        options=popular_stocks,
        default=["AAPL", "MSFT", "GOOGL"],
        max_selections=10
    )
    
    # 手動で銘柄を追加
    manual_ticker = st.text_input("追加の銘柄コードを入力（例: TSLA, NVDA）")
    if manual_ticker:
        additional_tickers = [ticker.strip().upper() for ticker in manual_ticker.split(',')]
        selected_tickers.extend(additional_tickers)
        selected_tickers = list(set(selected_tickers))  # 重複を除去
    
    if len(selected_tickers) > 0:
        st.info(f"選択された銘柄: {', '.join(selected_tickers)}")
        
        # 複数銘柄のデータ取得と比較チャート表示
        # （実装は以下で続行）
        
        # プレースホルダーメッセージ
        st.success("✅ 複数銘柄比較機能を実装中...")
        
        # 複数銘柄データの取得
        comparison_data = {}
        
        # 各銘柄のデータを取得
        for ticker in selected_tickers:
            with st.spinner(f"🔄 {ticker} のデータを取得中..."):
                try:
                    stock_data = get_stock_data(ticker, days)
                    if not stock_data.empty:
                        comparison_data[ticker] = stock_data
                        st.success(f"✅ {ticker}: {len(stock_data)} 日分のデータを取得")
                    else:
                        st.warning(f"⚠️ {ticker}: データが取得できませんでした")
                except Exception as e:
                    st.error(f"❌ {ticker}: エラー - {str(e)}")
        
        # 比較チャートの作成
        if comparison_data:
            # AI分析の表示（条件付き）
            if enable_ai_analysis:
                st.subheader("🤖 AI複数銘柄比較分析")
                
                # AI分析を実行
                with st.spinner("🤖 AIが複数銘柄を分析中..."):
                    ai_comparison_analysis = generate_ai_comparison_analysis(comparison_data, normalize_prices, selected_model)
                
                # AI分析結果を表示
                # 背景色を薄いブルーに設定
                st.markdown(f"""
                <div style="background-color: #E6F3FF; padding: 15px; border-radius: 10px; border-left: 4px solid #0066CC;">
                <h4 style="color: #0066CC; margin-bottom: 10px;">🤖 AI比較分析結果:</h4>
                <div style="color: #333333; line-height: 1.6;">
                {ai_comparison_analysis}
                </div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("---")
            
            create_comparison_chart(comparison_data, normalize_prices)
            create_comparison_table(comparison_data, normalize_prices)
    else:
        st.warning("⚠️ 比較する銘柄を選択してください")



# アプリケーションの実行
# Streamlitアプリは直接実行される
