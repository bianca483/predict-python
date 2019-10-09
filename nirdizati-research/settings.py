"""
Django settings for nirdizati-research project.

Generated by 'django-admin startproject' using Django 1.11.7.

For more information on this file, see
https://docs.djangoproject.com/en/1.11/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/1.11/ref/settings/
"""

import os

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/1.11/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'mt-0m%5!@ef9lcoga)nu@c9=ai@2_9l!*6v@u(^*zi-9w=882-'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = ['*']

# Application definition

INSTALLED_APPS = [
    'src.common.apps.CommonConfig',
    'src.logs.apps.LogsConfig',
    'src.split.apps.SplitConfig',
    'src.encoding.apps.EncodingConfig',
    'src.labelling.apps.LabellingConfig',
    'src.cache.apps.CacheConfig',
    'src.clustering.apps.ClusteringConfig',
    'src.predictive_model.apps.PredictiveModelConfig',
    'src.predictive_model.classification.apps.ClassificationConfig',
    'src.predictive_model.regression.apps.RegressionConfig',
    'src.predictive_model.time_series_prediction.apps.TimeSeriesPredictionConfig',
    'src.evaluation.apps.EvaluationConfig',
    'src.jobs.apps.JobsConfig',
    'src.hyperparameter_optimization.apps.HyperparameterOptimizationConfig',
    'src.runtime.apps.RuntimeConfig',

    'rest_framework',
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.messages',
    'django.contrib.sessions',
    'django.contrib.staticfiles',
    "django_rq",
    'corsheaders',
    'ws4redis',
    'django_extensions'
]

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '[%(asctime)s] %(levelname)s %(module)s %(message)s'
        }
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'verbose'
        }
    },
    'loggers': {
        'logs': {
            'handlers': ['console'],
            'level': 'INFO'
        }
    }
}

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'corsheaders.middleware.CorsMiddleware',
]

ROOT_URLCONF = 'nirdizati-research.urls'
CORS_ORIGIN_ALLOW_ALL = True

REST_FRAMEWORK = {
    # Use Django's standard `django.contrib.auth` permissions,
    # or allow read-only access for unauthenticated users.
    'DEFAULT_PERMISSION_CLASSES': [
        # 'rest_framework.permissions.DjangoModelPermissionsOrAnonReadOnly'
    ],
    'DEFAULT_RENDERER_CLASSES': (
        'src.jobs.json_renderer.PalJSONRenderer',
        'rest_framework.renderers.BrowsableAPIRenderer',
    )
}

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                'ws4redis.context_processors.default'
            ],
        },
    },
]

WSGI_APPLICATION = 'ws4redis.django_runserver.application'

# Database
# https://docs.djangoproject.com/en/1.11/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': os.environ["DB_NAME"],
        'USER': os.environ["DB_USER"],
        'PASSWORD': os.environ["DB_PASSWORD"],
        'HOST': os.environ.get("DB_HOST", "localhost"),
        'PORT': os.environ.get("DB_PORT", ""),
        'TEST': {
            'NAME': os.environ.get("DB_NAME_TEST", ""),
        }
    }
}




# Password validation
# https://docs.djangoproject.com/en/1.11/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Internationalization
# https://docs.djangoproject.com/en/1.11/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = False

USE_L10N = False

USE_TZ = True

STATIC_URL = '/static/'

RQ_QUEUES = {
    'default': {
        'HOST': os.environ["REDIS_HOST"],
        'PORT': os.environ["REDIS_PORT"],
        'DB': 0,
        'DEFAULT_TIMEOUT': 7200,
    }
}

WS4REDIS_CONNECTION = {
    'host': os.environ["REDIS_HOST"],
    'port': os.environ["REDIS_PORT"],
    'db': 0,
    'password': None,
}

WEBSOCKET_URL = '/ws/'
WS4REDIS_PREFIX = 'ws'
