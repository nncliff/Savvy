-- Minimal PostgreSQL trigger for bookmark notifications
CREATE OR REPLACE FUNCTION notify_bookmark_changes()
RETURNS TRIGGER AS $$
BEGIN
    PERFORM pg_notify('bookmark_changes', 'change');
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS bookmark_changes_trigger ON "bookmarkLinks";
CREATE TRIGGER bookmark_changes_trigger
    AFTER INSERT OR UPDATE OR DELETE ON "bookmarkLinks"
    FOR EACH ROW EXECUTE FUNCTION notify_bookmark_changes();
