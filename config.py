import os


class Config():
    TESTING = False
    SECRET_KEY = os.environ.get('SECRET_KEY', '5f352379324c22')
    CSRF_ENABLED = True
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'None'

class ProductionConfig(Config):
    DEBUG = False

class DevelopmentConfig(Config):
    ENV = "development"
    DEVELOPMENT = True
