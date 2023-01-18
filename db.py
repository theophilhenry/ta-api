import mysql.connector
import logging as logger
# https://medium.com/@emergeit/database-crud-with-fastapi-keep-it-simple-and-documented-with-python-and-mysql-db-7d626520c37a
class Database:
    connection = None
    def __init__(self):
        if self.connection is None or self.connection.is_connected() == False:
            self.connection = mysql.connector.connect(
                user="root",
                password="",
                host="localhost",
                database="ta_160419078",
            )
            self.connection.autocommit = False
    def query(self, query, autoCommit=None, fetch="ALL"):
        try:
            cursor = self.connection.cursor()
            result = cursor.execute(query)
            if autoCommit is not None:
                self.connection.commit()
                operation = True if cursor.lastrowid == 0 else {"id": cursor.lastrowid}
                return {"data": operation}
            
            fields = [field_md[0] for field_md in cursor.description]
            if fetch != "SINGLE":
                result = [dict(zip(fields, row)) for row in cursor.fetchall()]
                return {"data": result}
            else:
                result = [dict(zip(fields, row)) for row in cursor.fetchone()]
                return {"data": result}
        except Exception as e:
            logger.error(e)
            return {"data": None, "error": e, "query": query}