#!/usr/bin/env bash

pip install --upgrade pip

python manage.py collectstatic --noinput

python manage.py migrate