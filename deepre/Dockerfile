# PythonのDockerイメージをダウンロード
FROM python:3.12

# Pythonの出力表示をDocker用に設定
ENV PYTHONUNBUFFERED=1

# Set timezone to JST
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install cron
RUN apt-get update && apt-get install -y cron curl && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリを指定
WORKDIR /app

# pipを使ってpoetryをインストール
#RUN pip install poetry

# poetryの設定ファイルが存在するなら、コピーしてパッケージをインストール
#COPY pyproject.toml* poetry.lock* ./
#RUN if [ -f pyproject.toml ]; then poetry install --no-root; fi

# pipをアップグレード
RUN pip install --upgrade pip

# requirements.txtをコピーして依存関係をインストール
COPY requirements.txt ./
RUN pip install -r requirements.txt

# GraphRAG CLIが利用可能か確認
RUN python -m graphrag --help || echo "GraphRAG module installed but CLI might not be available"

# アプリケーションファイルをコピー
COPY . .

# Ensure scripts have proper permissions and verify they exist
RUN ls -la /app/start.sh /app/call_random_query.sh
RUN chmod +x /app/start.sh /app/call_random_query.sh
RUN ls -la /app/start.sh /app/call_random_query.sh

# Set up cron job for 6:33 AM and 11:33 PM JST
RUN echo "33 6 * * * /app/call_random_query.sh" > /etc/cron.d/random-query
RUN echo "33 23 * * * /app/call_random_query.sh" >> /etc/cron.d/random-query
RUN chmod 0644 /etc/cron.d/random-query
RUN crontab /etc/cron.d/random-query

# Verify the startup script is executable
RUN test -x /app/start.sh && echo "start.sh is executable" || echo "start.sh is NOT executable"

# Use the startup script as CMD
CMD ["/app/start.sh"]
