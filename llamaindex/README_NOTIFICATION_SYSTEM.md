# RAG API with Real-time Bookmark Notifications & Deep Research

This RAG API provides real-time bookmark indexing and advanced deep research capabilities using the Plan → Think → Action → Analyze pattern.

## Changes Made

### 1. Event-Driven Index Updates
- Replaced `update_index_periodically()` with `listen_for_bookmark_changes()`
- Uses PostgreSQL's LISTEN/NOTIFY mechanism for real-time notifications
- Updates the vector index immediately when bookmarks are added, updated, or deleted

### 2. New Dependencies
- Added `psycopg2` for PostgreSQL notifications (already in requirements.txt)
- Added logging for better debugging and monitoring

### 3. Deep Research Agent (NEW)
- Implements Plan → Think → Action → Analyze pattern for comprehensive research
- Uses local document search tool powered by LlamaIndex
- Provides structured research with detailed analysis

### 4. New Endpoints
- `POST /rebuild-index` - Manually trigger index rebuild
- `GET /health` - Health check with index status and document count
- `POST /deep-research` - Conduct deep research using structured pattern
- `GET /deep-research` - GET endpoint for deep research queries

## Setup Instructions

### 1. Database Setup
Run the SQL trigger setup script in your PostgreSQL database:

```bash
psql -h your_host -U your_user -d your_database -f setup_bookmark_trigger.sql
```

Or execute the SQL commands directly:
```sql
-- See setup_bookmark_trigger.sql for the complete script
```

### 2. Start the API
```bash
cd llamaindex
pip install -r requirements.txt
uvicorn rag_api:app --host 0.0.0.0 --port 8000
```

## How It Works

1. **Trigger Setup**: PostgreSQL trigger monitors the `bookmarkLinks` table
2. **Notifications**: When INSERT/UPDATE/DELETE occurs, trigger sends notification via `pg_notify()`
3. **Listener**: Python application listens for notifications using `psycopg2`
4. **Index Update**: When notification received, index is immediately rebuilt with latest data

## Benefits

- **Real-time Updates**: No delay between bookmark creation and index availability
- **Resource Efficient**: No unnecessary periodic polling
- **Scalable**: Only rebuilds when actual changes occur
- **Reliable**: Uses PostgreSQL's built-in notification system

## Testing

1. **Check Health**: `GET /health` - Verify the service is running and index is ready
2. **Manual Rebuild**: `POST /rebuild-index` - Force index rebuild for testing
3. **Add Bookmark**: Add a bookmark through your application and check logs for real-time updates
4. **Query**: Use existing `/query` endpoints to search the updated index

## Monitoring

The application logs important events:
- Initial connection and listener setup
- Notifications received from PostgreSQL
- Index rebuilds and document counts
- Any errors in the notification system

## Deep Research Functionality

### Overview
The deep research feature implements a structured approach to information gathering using the **Plan → Think → Action → Analyze** pattern:

1. **Plan**: Breaks down research topics into specific, focused questions
2. **Think**: Reasons about search strategies and keywords for each question  
3. **Action**: Executes searches using the local document index
4. **Analyze**: Synthesizes findings into comprehensive insights

### Using Deep Research

#### POST Request
```bash
curl -X POST "http://localhost:8000/deep-research" \
  -H "Content-Type: application/json" \
  -d '{"research_query": "artificial intelligence trends"}'
```

#### GET Request
```bash
curl "http://localhost:8000/deep-research?research_query=sustainable%20technology"
```

### Response Structure
```json
{
  "research_query": "your research topic",
  "plan": {
    "research_topic": "your research topic",
    "plan": [
      "Question 1: specific question",
      "Question 2: specific question", 
      "Question 3: specific question"
    ]
  },
  "findings": [
    {
      "question": "Question 1: specific question",
      "reasoning": "Search strategy explanation",
      "search_queries": ["query1", "query2", "query3"],
      "search_results": [
        {
          "query": "query1",
          "result": "search results and analysis"
        }
      ]
    }
  ],
  "analysis": "Comprehensive synthesis of all findings with insights, patterns, and recommendations"
}
```

### Example Research Topics
- "machine learning in healthcare"
- "remote work productivity tools"
- "sustainable energy technologies"
- "cybersecurity best practices"
- "blockchain applications"

### Testing Deep Research
Run the comprehensive test suite:
```bash
cd llamaindex
python test_deep_research.py
```

This will test:
- API connectivity
- Index status
- Basic query functionality
- Deep research POST endpoint
- Deep research GET endpoint

## Troubleshooting

- **Index not updating**: Check PostgreSQL logs and ensure trigger is installed
- **Connection issues**: Verify database credentials and network connectivity
- **Performance**: Monitor index rebuild frequency and consider batch updates for high-volume scenarios
- **Deep research errors**: Ensure OpenAI API key is configured and index is ready
