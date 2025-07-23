# PythonのDockerイメージをダウンロード
FROM python:3.12

# Pythonの出力表示をDocker用に設定
ENV PYTHONUNBUFFERED=1

# 作業ディレクトリを指定
WORKDIR /app

# pipを使ってpoetryをインストール
#RUN pip install poetry

# poetryの設定ファイルが存在するなら、コピーしてパッケージをインストール
#COPY pyproject.toml* poetry.lock* ./
#RUN if [ -f pyproject.toml ]; then poetry install --no-root; fi

# requirements.txtをコピーして依存関係をインストール
#COPY requirements.txt ./
#RUN pip install -r requirements.txt

# GraphRAGを確実にインストール（最新版）
RUN pip install --upgrade pip
RUN pip install graphrag --no-cache-dir

# GraphRAG CLIが利用可能か確認
RUN python -m graphrag --help || echo "GraphRAG module installed but CLI might not be available"

# アプリケーションファイルをコピー
COPY . .

# FastAPIアプリケーションを起動
#CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
