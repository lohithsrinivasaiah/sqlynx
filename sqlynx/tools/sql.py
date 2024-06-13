from __future__ import annotations

from llama_index.agent.openai import OpenAIAgent
from llama_index.core.indices.struct_store.sql_query import SQLTableRetrieverQueryEngine
from llama_index.core.response import Response
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from sqlalchemy.engine import Result
from sqlalchemy.exc import OperationalError

from sqlynx.datamodels import SQLResult
from sqlynx.engines.sql import SQLEngine


class SQLQueryTool(BaseToolSpec):
    """
    A tool designed for the
        1. `generate_sql_query`
        2. `execute_sql_query`
        3. `refine_sql_query`
    SQL queries based on user input.

    This tool provides functionality to generate SQL queries from user
    questions and execute them and return response and also refine if any errors in query.
    """

    spec_functions = ["generate_sql_query", "execute_sql_query"]

    def __init__(self):
        """
        Initializes the SQLQueryTool instance.
        """
        super().__init__()
        self.sql_engine: SQLEngine = SQLEngine()
        self.query_engine: SQLTableRetrieverQueryEngine = self.sql_engine.get_query_engine()

    def generate_sql_query(self, user_question: str) -> str:
        """
        Generate SQL query based on the user's question.

        Args:
            user_question (str): The question posed by the user,
            which serves as the basis for the SQL query.

        Returns:
            response (str): SQL Query generated based on user query.
        """
        self.user_question: str = user_question
        response: Response = self.query_engine.query(self.user_question)
        generated_sql_query: str = response.metadata.get("sql_query")
        return generated_sql_query

    def execute_sql_query(self, generated_sql_query: str) -> Result:
        """
        Execute the generated SQL query and return the results

        Args:
            generated_sql_query (str): The generated SQL query.

        Returns:
            result (Result): Result of executed sql query.
        """
        success: bool
        result: Result | OperationalError
        success, result = self.sql_engine.execute_query(generated_sql_query)
        normalized_result = self.normalize_result(result)
        if not success:
            self.refine_sql_query(self.user_question, generated_sql_query, normalized_result)
        return normalized_result

    def refine_sql_query(self, user_question: str, generated_sql_query: str, result: Result):
        """
        Refine the SQL query if it fails and re-execute it.

        Args:
            user_question (str): The original user question.
            generated_sql_query (str): The generated SQL query.
            result (Result): The result of the query execution.

        Returns:
            result (Result): The result of the refined and re-executed query.
        """
        new_query = self.generate_sql_query(user_question)
        success, result = self.sql_engine.execute_query(new_query)
        normalized_result = self.normalize_result(result)
        return normalized_result

    def normalize_result(self, result: Result) -> SQLResult:
        """
        Normalize the SQLAlchemy Result object into a clean form (list of dicts).

        Args:
            result (Result): The SQLAlchemy Result object.

        Returns:
            normalized_result (SQLResult): A list of dictionaries representing the rows.
        """
        if isinstance(result, OperationalError):
            error_message = str(result)
            return SQLResult(
                columns=[],
                data=[],
                metadata={
                    "is_visualizable": False,
                    "is_single_value": False,
                    "error": error_message,
                },
            )

        columns: list(str) = result.keys()
        data: list(tuple) = [tuple(row) for row in result.fetchall()]
        is_visualizable: bool = len(columns) > 1 or len(data) > 1
        is_single_value: bool = len(columns) == 1 and len(data) == 1

        return SQLResult(
            columns=columns,
            data=data,
            metadata={"is_visualizable": is_visualizable, "is_single_value": is_single_value},
        )


if __name__ == "__main__":
    sql_query_tool = SQLQueryTool().to_tool_list()
    agent = OpenAIAgent.from_tools(sql_query_tool, verbose=True)
    while True:
        user_question = input("Your Question: ")
        response = agent.chat(user_question)
        print(response)
