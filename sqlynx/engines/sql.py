from __future__ import annotations

import os
from abc import abstractmethod
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import SQLDatabase
from llama_index.core import VectorStoreIndex
from llama_index.core.indices.struct_store.sql_query import SQLTableRetrieverQueryEngine
from llama_index.core.objects import ObjectIndex
from llama_index.core.objects import SQLTableNodeMapping
from llama_index.core.objects import SQLTableSchema
from sqlalchemy import create_engine
from sqlalchemy import MetaData
from sqlalchemy.engine import Engine
from sqlalchemy.engine import Result
from sqlalchemy.exc import OperationalError
from sqlalchemy.sql import text

from sqlynx.utils.exceptions import DatabaseConnectionError
from sqlynx.utils.exceptions import MissingEnvVarError

load_dotenv()

INDEX_DIRECTORY: Path = Path("storage/sql_index_data")


class SQLEngine:
    """
    Manages SQL database connections and operations.
    Supports MySQL and PostgreSQL.
    """

    REQUIRED_ENV_VARS = [
        "DB_SCHEME",
        "DB_NAME",
        "DB_USER",
        "DB_PASSWORD",
        "DB_HOST",
    ]

    DB_MODULES = {
        "mysql": ("pymysql", "mysql+pymysql", "3306"),
        "postgresql": ("psycopg2", "postgresql+psycopg2", "5432"),
    }

    def __init__(self, include_tables: list[str] = None, similarity_top_k: int = 5) -> None:
        """
        Initialize the SQLEngine instance by loading environment variables.
        """
        self._load_env_vars()
        self._create_engine()
        self._sql_database = SQLDatabase(self.engine)
        self.meta = MetaData()
        self.meta.reflect(bind=self.engine)

        self._create_table_objects_and_mappings(include_tables)
        self._init_index()
        self.query_engine = self._create_query_engine()

    def _load_env_vars(self) -> None:
        """
        Load required environment variables.

        Raises:
            MissingEnvVarError: If any required environment variables are missing.
        """
        missing_env_vars = [var for var in self.REQUIRED_ENV_VARS if os.getenv(var) is None]

        if missing_env_vars:
            raise MissingEnvVarError(", ".join(missing_env_vars))

        self.db_host = os.getenv("DB_HOST")
        self.db_name = os.getenv("DB_NAME")
        self.db_password = os.getenv("DB_PASSWORD")
        self.db_port = os.getenv("DB_PORT")
        self.db_scheme = os.getenv("DB_SCHEME")
        self.db_user = os.getenv("DB_USER")

    @abstractmethod
    def from_uri(self, uri: str) -> SQLTableRetrieverQueryEngine:
        """
        Create an SQLTableRetrieverQueryEngine instance from a given URI.

        Args:
            uri (str): The database URI.

        Returns:
            SQLTableRetrieverQueryEngine: An instance of SQLTableRetrieverQueryEngine.
        """

    def _build_uri(self) -> str:
        """
        Build the database URI from the loaded environment variables.

        Returns:
            str: The constructed database URI.

        Raises:
            ValueError: If the database scheme is unsupported.
            ImportError: If the necessary database driver is not installed.
        """
        if self.db_scheme not in self.DB_MODULES:
            raise ValueError(
                "Unsupported database scheme. Supported schemes are `mysql` and `postgresql`.",
            )

        module_name, db_url_prefix, default_port = self.DB_MODULES[self.db_scheme]
        self._ensure_module_installed(module_name)
        self.db_port = self.db_port or default_port

        return (
            f"{db_url_prefix}://{self.db_user}:"
            f"{self.db_password}@{self.db_host}:"
            f"{self.db_port}/{self.db_name}"
        )

    def _ensure_module_installed(self, module_name: str) -> None:
        """
        Ensure that the required database driver module is installed.

        Args:
            module_name (str): The name of the database driver module.

        Raises:
            ImportError: If the module is not installed.
        """
        try:
            __import__(module_name)
        except ImportError:
            raise ImportError(
                f"{module_name} is not installed. Install it using `pip install {module_name}`.",
            )

    def _create_engine(self) -> Engine:
        """
        Create a SQLAlchemy Engine.

        Returns:
            Engine: The SQLAlchemy Engine instance.

        Raises:
            DatabaseConnectionError: If the connection to the database fails.
        """
        self.uri = self._build_uri()
        self.engine = create_engine(self.uri)

        if self._check_db_connection():
            return self.engine
        else:
            raise DatabaseConnectionError("Failed to connect to the database.")

    def _check_db_connection(self) -> bool:
        """
        Check the connection to the database.

        Returns:
            bool: True if the connection is successful, False otherwise.
        """
        try:
            with self.engine.connect():
                return True
        except OperationalError:
            return False

    def _create_table_objects_and_mappings(
        self,
        include_table: list[str] = None,
    ) -> None:
        """
        Create table schema objects and table node mappings.

        Args:
            include_table (list[str]): list of table names to include.
        """
        if include_table is None:
            include_table = self.meta.tables.keys()

        self.table_schema_objects = [
            SQLTableSchema(table_name=table_name) for table_name in include_table
        ]
        self.table_node_mapping = SQLTableNodeMapping(self._sql_database)

    def _init_index(self) -> None:
        """
        Initialize index.
        """
        try:
            self.object_index = ObjectIndex.from_persist_dir(
                persist_dir=INDEX_DIRECTORY,
                object_node_mapping=self.table_node_mapping,
            )
        except FileNotFoundError:
            self.object_index = ObjectIndex.from_objects(
                objects=self.table_schema_objects,
                object_mapping=self.table_node_mapping,
                index_cls=VectorStoreIndex,
            )
            self.object_index.persist(persist_dir=INDEX_DIRECTORY)

    def _create_query_engine(self) -> SQLTableRetrieverQueryEngine:
        """
        Create a SQL query engine.

        Returns:
            SQLTableRetrieverQueryEngine: The created query engine.
        """
        return SQLTableRetrieverQueryEngine(
            sql_database=self._sql_database,
            table_retriever=self.object_index.as_retriever(similarity_top_k=5),
            sql_only=True,
        )

    def get_query_engine(self) -> SQLTableRetrieverQueryEngine:
        """
        Get the SQL query engine.

        Returns:
            SQLTableRetrieverQueryEngine: The query engine.
        """
        return self.query_engine

    def execute_query(self, sql_query: str) -> tuple[bool, Result | OperationalError]:
        """
        Execute the given SQL query and return the results.

        Args:
            sql_query (str): The SQL query to execute.

        Returns:
            tuple: A tuple containing a boolean indicating the success of the
                   query execution and the result set if successful (as a Result object),
                   or an error if unsuccessful.
        """
        try:
            with self.engine.connect() as connection:
                result: Result = connection.execute(text(sql_query))
                return True, result
        except OperationalError as error:
            return False, error
