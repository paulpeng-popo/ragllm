import chromadb
import pymysql
import pymysql.cursors

from hashlib import md5
from typing import List
from tools.translator import translate
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cuda"}
)


class QuestionCluster:
    def __init__(
        self,
        collection_name: str = "questions",
        database_name: str = "clustering",
    ):
        chroma_client = chromadb.HttpClient(
            database=database_name,
        )
        self.vectorstore = Chroma(
            client=chroma_client,
            embedding_function=EMBEDDINGS,
            collection_name=collection_name,
        )
    
    def add_question(self, question: str) -> None:
        """Add a question to the question cluster.

        Args:
            question (str): The question to add.
        """
        # english_question = translate(question)
        self.vectorstore.add_documents(
            documents=[Document(
                metadata={"chinese": question},
                page_content=question
            )],
            ids=[md5(question.encode()).hexdigest()]
        )
        
    def get_similar_question(
        self,
        question: str,
        distance_threshold: float = 0.2,
        top_k: int = 3
    ) -> str:
        """Get the most similar question to the given question.

        Args:
            question (str): The question to find the most similar question to.
            distance_threshold (float, optional): The distance threshold to consider a question similar. Defaults to 0.2.
            top_k (int, optional): The number of similar questions to return. Defaults to 3.

        Returns:
            str: The most similar question to the given question.
        """
        # english_question = translate(question)
        results = self.vectorstore.similarity_search_with_score(
            query=question,
            k=top_k
        )
        # print("Search Ranking:")
        # for result, score in results:
        #     print(result.metadata["chinese"], score)
        if results:
            result, score = results[0]
            if score < distance_threshold:
                return result.metadata["chinese"]
        # add the question to the cluster if no similar question is found
        self.add_question(question)
        return question
    
    def get_similar_questions(
        self,
        question: str,
        distance_threshold: float = 0.2,
        top_k: int = 3
    ) -> List[str]:
        """Get the most similar questions to the given question.

        Args:
            question (str): The question to find the most similar questions to.
            distance_threshold (float, optional): The distance threshold to consider a question similar. Defaults to 0.2.
            top_k (int, optional): The number of similar questions to return. Defaults to 3.

        Returns:
            List[str]: The most similar questions to the given question.
        """
        results = self.vectorstore.similarity_search_with_score(
            query=question,
            k=top_k
        )
        if results:
            return [result.metadata["chinese"] for result, score in results if score < distance_threshold]
        return []


class Database:
    def __init__(self, database_name: str = "rag_database"):
        self.connection = pymysql.connect(
            host="140.116.245.154",
            port=3306,
            user="root",
            password="wmmks65802",
            database=database_name,
            charset="utf8"
        )
        
    def check_table_exist(self, table_name: str) -> bool:
        """Check if a table exists in the database.

        Args:
            table_name (str): The name of the table to check.

        Returns:
            bool: True if the table exists, False otherwise.
        """
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(f"SHOW TABLES LIKE '{table_name}'")
                return cursor.fetchone() is not None
        except Exception as e:
            print(e)
            return False
        
    def create_table(self, table_name: str, columns: List[str]) -> bool:
        """Create a table in the database with the given columns.

        Args:
            table_name (str): The name of the table to create.
            columns (List[str]): The columns to create in the table.

        Returns:
            bool: True if the table was created, False otherwise.
        """
        if self.check_table_exist(table_name):
            return False
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    f"CREATE TABLE {table_name} ({', '.join(columns)})"
                )
            return True
        except Exception as e:
            print(e)
            return False
            
    def insert(self, table_name: str, columns: List[str], values: List) -> int:
        """Insert a row into a table in the database.

        Args:
            table_name (str): The name of the table to insert into.
            columns (List[str]): The columns to insert into.
            values (List[Any]): The values to insert into the table.

        Returns:
            int: The rowid of the inserted row. If the row insertion failed, -1 is returned.
        """
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({', '.join(['%s']*len(values))})",
                    values
                )
                self.connection.commit()
                return cursor.lastrowid
        except Exception as e:
            print(e)
            return -1
        
    def select(self, table_name: str, columns: List[str], where: str = "") -> List:
        """Select rows from a table in the database.

        Args:
            table_name (str): The name of the table to select from.
            columns (List[str]): The columns to select.
            where (str): The WHERE clause to use in the query.

        Returns:
            list: The selected rows as a list of dictionaries.
        """
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT {', '.join(columns)} FROM {table_name}" + (f" WHERE {where}" if where else "")
                )
                return cursor.fetchall()
        except Exception as e:
            print(e)
            return []
        
    def close(self) -> None:
        """Close the connection
        """
        self.connection.close()
        
        
class Feedbacks(Database):
    def __init__(self):
        super().__init__()
        
        self.table_name = "feedbacks"
        self.create_table(
            self.table_name,
            [
                "user_query varchar(255) primary key",
                "response text",
                "good_count int",
                "bad_count int",
                "feedback text"
            ]
        )
        self.qc = QuestionCluster()
            
    def insert_feedback(self, user_query: str, response: str, good_count: int = 0, bad_count: int = 0, feedback: str = "") -> bool:
        """Insert a feedback into the feedbacks table.

        Args:
            user_query (str): The user query to insert.
            response (str): The response to insert.
            good_count (int): The good count to insert.
            bad_count (int): The bad count to insert.
            feedback (str): The feedback to insert.

        Returns:
            bool: True if the feedback was inserted, False otherwise.
        """
        user_query = self.qc.get_similar_question(user_query)
        print("Similar question:", user_query)
        
        # check if the user_query already exists
        row = self.get_feedback(user_query)
        
        if row:
            # print("Find similar question, combining the feedbacks...")
            # combine the old values with the new values
            # response = row[1] + "\n" + response
            good_count += row[2]
            bad_count += row[3]
            feedback = row[4] + "\n" + feedback
            return self.update_feedback(user_query, response, good_count, bad_count, feedback)
        else:
            return self.insert(
                self.table_name,
                ["user_query", "response", "good_count", "bad_count", "feedback"],
                [user_query, response, good_count, bad_count, feedback]
            ) != -1

    def update_feedback(self, user_query: str, response: str, good_count: int = 0, bad_count: int = 0, feedback: str = "") -> bool:
        """Update a feedback in the feedbacks table.

        Args:
            user_query (str): The user query to update.
            response (str): The response to update.
            good_count (int): The good count to update.
            bad_count (int): The bad count to update.
            feedback (str): The feedback to update.

        Returns:
            bool: True if the feedback was updated, False otherwise.
        """
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    f"UPDATE {self.table_name} SET response=%s, good_count=%s, bad_count=%s, feedback=%s WHERE user_query=%s",
                    [response, good_count, bad_count, feedback, user_query]
                )
            self.connection.commit()
            return True
        except Exception as e:
            print(e)
            return False
        
    def get_feedback(self, user_query: str) -> List:
        """Get a feedback from the feedbacks table.

        Args:
            user_query (str): The user query to get the feedback for.

        Returns:
            List: The feedback as a list of values.
        """
        res = self.select(
            self.table_name,
            ["user_query", "response", "good_count", "bad_count", "feedback"],
            f"user_query='{user_query}'"
        )
        return res[0] if res else []
    
    def get_relevant_feedbacks(self, user_query: str) -> List:
        """Get relevant feedbacks from the feedbacks table.

        Args:
            user_query (str): The user query to get the relevant feedbacks for.

        Returns:
            List: The relevant feedbacks as a list of values.
        """
        qc = QuestionCluster()
        similar_questions = qc.get_similar_questions(user_query)
        feedbacks = []
        for q in similar_questions:
            feedback = self.get_feedback(q)
            if feedback:
                feedbacks.append(feedback)
        return feedbacks
    
    def get_all_feedbacks(self) -> List:
        """Get all feedbacks from the feedbacks table.

        Returns:
            List: All feedbacks as a list of values.
        """
        return self.select(
            self.table_name,
            ["user_query", "response", "good_count", "bad_count", "feedback"]
        )
        

class CollectionNames(Database):
    def __init__(self):
        super().__init__()
        
        self.table_name = "collection_names"
        self.create_table(
            self.table_name,
            [
                "name_md5 varchar(255) primary key",
                "collection_name text"
            ]
        )
            
    def insert_collection_name(self, name_md5: str, collection_name: str) -> bool:
        """Insert a collection name into the collection_names table.

        Args:
            name_md5 (str): The name MD5 to insert.
            collection_name (str): The collection name to insert.

        Returns:
            bool: True if the collection name was inserted, False otherwise.
        """
        # check if the name_md5 already exists
        if self.select(self.table_name, ["name_md5"], f"name_md5='{name_md5}'"):
            return False
        return self.insert(
            self.table_name,
            ["name_md5", "collection_name"],
            [name_md5, collection_name]
        ) != -1
        
    def get_name_by_hash(self, name_md5: str) -> str:
        """Get a collection name from the collection_names table.

        Args:
            name_md5 (str): The name MD5 to get the collection name for.

        Returns:
            str: The collection name.
        """
        res = self.select(
            self.table_name,
            ["collection_name"],
            f"name_md5='{name_md5}'"
        )
        return res[0][0] if res else ""
    
    def get_hash_by_name(self, collection_name: str) -> str:
        """Get a collection name from the collection_names table.

        Args:
            collection_name (str): The collection name to get the name MD5 for.

        Returns:
            str: The name MD5.
        """
        res = self.select(
            self.table_name,
            ["name_md5"],
            f"collection_name='{collection_name}'"
        )
        return res[0][0] if res else ""
        
        
class DocumentIDs(Database):
    def __init__(self):
        super().__init__()
        
        self.table_name = "document_ids"
        self.create_table(
            self.table_name,
            [
                "doc_id varchar(255) primary key",
                "file_name text"
            ]
        )
            
    def insert_document_id(self, doc_id: str, file_name: str) -> bool:
        """Insert a document ID into the document_ids table.

        Args:
            doc_id (str): The document ID to insert.
            file_name (str): The file name to insert.

        Returns:
            bool: True if the document ID was inserted, False otherwise.
        """
        # check if the doc_id already exists
        if self.select(self.table_name, ["doc_id"], f"doc_id='{doc_id}'"):
            return False
        return self.insert(
            self.table_name,
            ["doc_id", "file_name"],
            [doc_id, file_name]
        ) != -1
        
    def get_filename_by_id(self, doc_id: str) -> str:
        """Get a file name from the document_ids table.

        Args:
            doc_id (str): The document ID to get the file name for.

        Returns:
            str: The file name.
        """
        res = self.select(
            self.table_name,
            ["file_name"],
            f"doc_id='{doc_id}'"
        )
        return res[0][0] if res else ""
    
    def get_ids_by_filename(self, file_name: str) -> List[str]:
        """Get a list of document IDs from the document_ids table.

        Args:
            file_name (str): The file name to get the document IDs for.

        Returns:
            List[str]: The list of document IDs.
        """
        res = self.select(
            self.table_name,
            ["doc_id"],
            f"file_name='{file_name}'"
        )
        return [r[0] for r in res]
        
        
if __name__ == "__main__":
    feedbacks = Feedbacks()
    collection_names = CollectionNames()
    document_ids = DocumentIDs()
    
    qc = QuestionCluster()
    qc.add_question("什麼情況適合使用駝人影像式插管組？")
    qc.add_question("什麼情況不適合使用駝人影像式插管組？")
    qc.add_question("駝人影像式插管組的運作原理？")
    qc.add_question("駝人影像式插管組包含哪些主要零件？")
    qc.add_question("硬式影像探條的操作步驟？")
    qc.add_question("彎式影像喉頭鏡的操作步驟？")
    qc.add_question("駝人影像式插管組的維修說明？")
    qc.add_question("駝人影像式插管組螢幕無法運作的原因？")
    
    questions = [
        "什麼情況適合使用駝人影像式插管組？",
        "什麼情況不適合使用駝人影像式插管組？",
        "駝人影像式插管組的運作原理？",
        "駝人影像式插管組包含哪些主要零件？",
        "硬式影像探條的操作步驟？",
        "彎式影像喉頭鏡的操作步驟？",
        "駝人影像式插管組的維修說明？",
        "駝人影像式插管組螢幕無法運作的原因？",
        "駝人影像插管的螢幕不會亮是為什麼？",
        "駝人影像插管維修說明？",
        "如何使用彎式影像喉頭鏡？",
        "如何使用硬式影像探條？",
        "駝人影像式插管組有什麼零件？",
        "駝人影像插管組的工作原理？",
        "駝人影像式插管組的禁忌症？",
        "駝人影像插管組的適應症？",
    ]
    
    # test insertion
    for q in questions:
        print("Inserting feedback for:", q)
        feedbacks.insert_feedback(q, "", 0, 0, "")
    
    feedbacks.close()
    collection_names.close()
    document_ids.close()
