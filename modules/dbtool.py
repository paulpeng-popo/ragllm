import sqlite3
import traceback

from typing import List, Tuple, Dict, Any, Optional, Union


TABLES_CONFIG = {
    "feedbacks": {
        "columns": ["user_query", "response", "good_count", "bad_count", "feedback"],
        "column_types": ["TEXT", "TEXT", "INTEGER", "INTEGER", "TEXT"],
        "unique_columns": ["user_query"],
    },
}


class DBTool:
    def __init__(
        self,
        db_name: str = "db.sqlite"
    ) -> None:
        
        """Initialize the database connection and create tables if they do not already exist.
        
        Args:
            db_name (str): The name of the database file to connect to.
        """
        
        # ensure variable TABLES_CONFIG exists and is valid
        assert "TABLES_CONFIG" in globals(), "TABLES_CONFIG must be defined"
        assert all(
            set(table_config.keys()) == {"columns", "column_types", "unique_columns"}
            for table_config in TABLES_CONFIG.values()
        ), "TABLES_CONFIG must have columns, column_types, and unique_columns keys"
        
        self.db_name = db_name
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        
        # create tables
        for table_name, table_config in TABLES_CONFIG.items():
            self._create_table(
                table_name,
                table_config["columns"],
                table_config["column_types"]
            )

    def _create_table(
        self,
        table_name: str,
        columns: List[str],
        column_types: List[str]
    ) -> str:
        
        """Create a table in the database with the given columns and column types, if it does not already exist.

        Args:
            table_name (str): The name of the table to create.
            columns (List[str]): The names of the columns in the table.
            column_types (List[str]): The types of the columns in the table.
            
        Returns:
            str: The name of the table that was created. If the table already exists, an empty string is returned. If an error occurred, -1 is returned.
        """
        
        # check if table exists
        if self.cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'").fetchone():
            return ""
        
        # check if columns and column_types have the same length
        assert len(columns) == len(column_types), "mismatched columns and column_types"
        
        columns = [f"{columns[i]} {column_types[i]}" for i in range(len(columns))]
        columns_str = ', '.join(columns)
        
        try:
            self.cursor.execute(f"CREATE TABLE {table_name} ({columns_str})")
        except Exception as e:
            traceback.print_exc()
            print(f"Error creating table {table_name}: {e}")
            return "-1"
            
        return table_name

    def insert(
        self,
        table_name: str,
        values: List[Any],
    ) -> int:
        
        """Insert a row into the table with the given values.
        If the row already exists (based on the unique columns), update the row instead.

        Args:
            table_name (str): The name of the table to insert into.
            values (List[Any]): The values to insert into the table.

        Returns:
            int: The rowid of the target row, or -1 if the row insertion failed.
        """
        
        assert table_name in TABLES_CONFIG, f"table {table_name} not in TABLES_CONFIG"
        
        table_config = TABLES_CONFIG[table_name]
        unique_columns = table_config["unique_columns"]
        
        columns_str = ', '.join(table_config["columns"])
        values_str = ', '.join(['?' for _ in table_config["columns"]])
        unique_values = [values[table_config["columns"].index(column)] for column in unique_columns]
        where_str = ' AND '.join([f"{unique_columns[i]} = ?" for i in range(len(unique_columns))])
        
        # check if row already exists
        row = self.cursor.execute(f"SELECT rowid FROM {table_name} WHERE {where_str}", unique_values).fetchone()
        
        if row:
            rowid = row[0]
            if table_name == "feedbacks":
                row = self.cursor.execute(f"SELECT good_count, bad_count, feedback FROM {table_name} WHERE rowid = ?", [rowid]).fetchone()
                # update good_count and bad_count
                values[table_config["columns"].index("good_count")] += row[0]
                values[table_config["columns"].index("bad_count")] += row[1]
                # concatenate feedback
                values[table_config["columns"].index("feedback")] = row[2] + "\n" + values[table_config["columns"].index("feedback")]
            self.update(table_name, rowid, table_config["columns"], values)
            return rowid
        
        try:
            self.cursor.execute(f"INSERT INTO {table_name} ({columns_str}) VALUES ({values_str})", values)
            rowid = self.cursor.lastrowid
            self.commit()
            return rowid
        except Exception as e:
            traceback.print_exc()
            print(f"Error inserting row into table {table_name}: {e}")
            return -1

    def update(
        self,
        table_name: str,
        row_id: int,
        columns: List[str],
        values: List[Any]
    ) -> None:
        
        """Update a row in the table with the given values.

        Args:
            table_name (str): The name of the table to update.
            row_id (int): The rowid of the row to update.
            columns (List[str]): The names of the columns to update.
            values (List[Any]): The values to update in the table.
        """
        
        assert table_name in TABLES_CONFIG, f"table {table_name} not in TABLES_CONFIG"
        
        columns_str = ', '.join([f"{columns[i]} = ?" for i in range(len(columns))])
        
        try:
            self.cursor.execute(f"UPDATE {table_name} SET {columns_str} WHERE rowid = ?", values + [row_id])
            self.commit()
        except Exception as e:
            traceback.print_exc()
            print(f"Error updating row in table {table_name}: {e}")
            
    def select_data(
        self,
        table_name: str,
        columns: List[str] = [],
        where: Optional[str] = None
    ) -> List[Tuple]:
        
        """Select data from the table with the given columns and where clause.

        Args:
            table_name (str): The name of the table to select from.
            columns (List[str]): The names of the columns to select. Empty list means select all columns.
            where (Optional[str]): The where clause to filter rows.

        Returns:
            List[Tuple]: The selected rows from the table.
        """
        
        assert table_name in TABLES_CONFIG, f"table {table_name} not in TABLES_CONFIG"
        
        columns_str = ', '.join(columns) if columns else "*"
        where_str = f" WHERE {where}" if where else ""
        
        # also need select rowid
        columns_str = "rowid, " + columns_str
        
        try:
            rows = self.cursor.execute(f"SELECT {columns_str} FROM {table_name}{where_str}").fetchall()
            return rows
        except Exception as e:
            traceback.print_exc()
            print(f"Error selecting data from table {table_name}: {e}")
            return []

    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    db = DBTool("../data/db.sqlite")
    db.update("feedbacks", 10, ["response"], ["new response"])
