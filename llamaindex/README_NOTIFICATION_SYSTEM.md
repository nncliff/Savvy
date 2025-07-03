# RAG API with Real-time Bookmark Notifications

This RAG API has been modified to update the LlamaIndex in real-time when new bookmark entities are added to PostgreSQL, instead of using periodic updates.

## Changes Made

### 1. Event-Driven Index Updates
- Replaced `update_index_periodically()` with `listen_for_bookmark_changes()`
- Uses PostgreSQL's LISTEN/NOTIFY mechanism for real-time notifications
- Updates the vector index immediately when bookmarks are added, updated, or deleted

### 2. New Dependencies
- Added `psycopg2` for PostgreSQL notifications (already in requirements.txt)
- Added logging for better debugging and monitoring

### 3. New Endpoints
- `POST /rebuild-index` - Manually trigger index rebuild
- `GET /health` - Health check with index status and document count

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

## Troubleshooting

- **Index not updating**: Check PostgreSQL logs and ensure trigger is installed
- **Connection issues**: Verify database credentials and network connectivity
- **Performance**: Monitor index rebuild frequency and consider batch updates for high-volume scenarios
