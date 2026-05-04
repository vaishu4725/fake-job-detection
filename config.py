"""config.py — TrueHire Backend Configuration"""
import os

class Config:
    SECRET_KEY      = os.environ.get('SECRET_KEY', 'truehire-secret-change-in-production-2025')
    MYSQL_HOST      = os.environ.get('MYSQL_HOST', 'localhost')
    MYSQL_USER      = os.environ.get('MYSQL_USER', 'root')
    MYSQL_PASSWORD  = os.environ.get('MYSQL_PASSWORD', 'your_password')
    MYSQL_DATABASE  = os.environ.get('MYSQL_DATABASE', 'truehire_db')
    MYSQL_PORT      = int(os.environ.get('MYSQL_PORT', 3306))
    DEBUG           = os.environ.get('FLASK_DEBUG', 'True') == 'True'
