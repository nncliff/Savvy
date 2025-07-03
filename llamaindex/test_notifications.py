#!/usr/bin/env python3
"""
Test script for the bookmark notification system.
This script can be used to verify that the LISTEN/NOTIFY setup is working correctly.
"""

import os
import psycopg2
import psycopg2.extensions
import time
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    """Get raw psycopg2 connection for LISTEN/NOTIFY"""
    db_host = os.getenv('POSTGRES_HOST', 'postgres')
    db_port = os.getenv('POSTGRES_PORT', '5432')
    db_name = os.getenv('POSTGRES_DB', 'karakeep')
    db_user = os.getenv('POSTGRES_USER', 'karakeep')
    db_password = os.getenv('POSTGRES_PASSWORD', 'karakeep_password')
    
    return psycopg2.connect(
        host=db_host,
        port=db_port,
        database=db_name,
        user=db_user,
        password=db_password
    )

def test_listener():
    """Test the PostgreSQL LISTEN/NOTIFY functionality"""
    print("Testing PostgreSQL notification system...")
    
    try:
        conn = get_db_connection()
        conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
        
        with conn.cursor() as cursor:
            # Listen for bookmark notifications
            cursor.execute("LISTEN bookmark_changes;")
            print("✓ Successfully listening for bookmark_changes notifications")
            print("Now add, update, or delete a bookmark in your database to test...")
            print("Press Ctrl+C to stop\n")
            
            while True:
                # Wait for notifications
                if conn.poll() == psycopg2.extensions.POLL_OK:
                    while conn.notifies:
                        notify = conn.notifies.pop(0)
                        print(f"📢 Notification received:")
                        print(f"   Channel: {notify.channel}")
                        print(f"   Payload: {notify.payload}")
                        print(f"   PID: {notify.pid}\n")
                
                time.sleep(0.5)  # Small delay to prevent busy waiting
                
    except KeyboardInterrupt:
        print("\n👋 Stopping listener...")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        try:
            conn.close()
            print("✓ Connection closed")
        except:
            pass

def test_trigger_exists():
    """Check if the trigger is properly installed"""
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT trigger_name, event_manipulation, event_object_table 
                FROM information_schema.triggers 
                WHERE trigger_name = 'bookmark_changes_trigger'
            """)
            
            result = cursor.fetchone()
            if result:
                print("✓ Trigger 'bookmark_changes_trigger' found:")
                print(f"   Name: {result[0]}")
                print(f"   Events: {result[1]}")
                print(f"   Table: {result[2]}")
                return True
            else:
                print("❌ Trigger 'bookmark_changes_trigger' not found!")
                print("Please run the setup_bookmark_trigger.sql script first.")
                return False
                
    except Exception as e:
        print(f"❌ Error checking trigger: {e}")
        return False
    finally:
        try:
            conn.close()
        except:
            pass

if __name__ == "__main__":
    print("🔧 Bookmark Notification System Test\n")
    
    # First check if trigger exists
    if test_trigger_exists():
        print()
        test_listener()
    else:
        print("\n💡 Setup instructions:")
        print("1. Run: psql -h your_host -U your_user -d your_database -f setup_bookmark_trigger.sql")
        print("2. Or execute the SQL commands in setup_bookmark_trigger.sql manually")
