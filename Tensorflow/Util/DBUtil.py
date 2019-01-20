# -*- coding: utf-8 -*-
import pymysql as pm

class DBUtil():

    def getConnect(self):
        conn = pm.connect(host="cdh01", user="root", password="adminal", database="al")
        return conn

    def executeSql(self,sqls,cursor,conn):
        if(type(sqls) == list):
            for sql in sqls:
                cursor.execute(sql)
        else:
            cursor.execute(sqls)
        conn.commit()

    def close(self,cursor,conn):
        cursor.close()
        conn.close()

    def runSql(self,sqls):
        conn = self.getConnect()
        cursor = conn.cursor()
        cursor.execute('SET NAMES UTF8')
        self.executeSql(sqls, cursor, conn)
        self.close(cursor, conn)