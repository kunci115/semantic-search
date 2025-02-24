"""
Database Factory to handle multiple database backends.
"""
import sqlite3
import redis
import psycopg2
import mysql.connector
from pymongo import MongoClient


class DatabaseFactory:
    """
    Factory class to create database instances dynamically.
    """

    @staticmethod
    def create_database(db_type="mongodb", **kwargs):
        """
        Creates a database instance based on the selected backend.

        Args:
            db_type (str): The backend type. Supported:
                "mongodb", "sqlite", "redis", "postgres", "mysql".
            **kwargs: Additional database connection parameters.

        Returns:
            A database connection/collection/instance, depending on the backend.
        """
        if db_type == "mongodb":
            client = MongoClient(kwargs.get("mongo_uri", "mongodb://localhost:27017/"))
            db_name = kwargs.get("db_name", "semantic_db")
            collection_name = kwargs.get("collection_name", "documents")
            return client[db_name][collection_name]

        if db_type == "sqlite":
            conn = sqlite3.connect(kwargs.get("db_path", "semantic_search.db"))
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc TEXT,
                    embedding TEXT
                )
            """)
            conn.commit()
            return conn

        if db_type == "redis":
            return redis.StrictRedis(
                host=kwargs.get("host", "localhost"),
                port=kwargs.get("port", 6379),
                db=kwargs.get("db", 0),
                decode_responses=True
            )

        if db_type == "postgres":
            conn = psycopg2.connect(
                dbname=kwargs.get("dbname", "semantic_db"),
                user=kwargs.get("user", "user"),
                password=kwargs.get("password", "password"),
                host=kwargs.get("host", "localhost"),
                port=kwargs.get("port", "5432")
            )
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    doc TEXT,
                    embedding TEXT
                )
            """)
            conn.commit()
            return conn

        if db_type == "mysql":
            conn = mysql.connector.connect(
                database=kwargs.get("database", "semantic_db"),
                user=kwargs.get("user", "user"),
                password=kwargs.get("password", "password"),
                host=kwargs.get("host", "localhost"),
                port=kwargs.get("port", 3306)
            )
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    doc TEXT,
                    embedding TEXT
                )
            """)
            conn.commit()
            return conn

        raise ValueError("Unsupported database type.")
