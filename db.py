"""db.py — MySQL connection helper using flask g context"""
import mysql.connector
from flask import g, current_app

def get_db():
    if 'db' not in g:
        cfg = current_app.config
        g.db = mysql.connector.connect(
            host=cfg['MYSQL_HOST'],
            user=cfg['MYSQL_USER'],
            password=cfg['MYSQL_PASSWORD'],
            database=cfg['MYSQL_DATABASE'],
            port=cfg['MYSQL_PORT'],
            autocommit=False,
            charset='utf8mb4'
        )
    return g.db

def close_db(e=None):
    db = g.pop('db', None)
    if db is not None and db.is_connected():
        db.close()
