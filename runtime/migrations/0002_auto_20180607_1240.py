# -*- coding: utf-8 -*-
# Generated by Django 1.11.7 on 2018-06-07 12:40
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('runtime', '0001_squashed_0009_xtrace_completed'),
    ]

    operations = [
        migrations.AddField(
            model_name='xevent',
            name='xid',
            field=models.CharField(default='', max_length=200),
        ),
        migrations.AddField(
            model_name='xtrace',
            name='first_event',
            field=models.DateTimeField(auto_now=True),
        ),
        migrations.AddField(
            model_name='xtrace',
            name='last_event',
            field=models.DateTimeField(auto_now=True),
        ),
        migrations.AddField(
            model_name='xtrace',
            name='n_events',
            field=models.IntegerField(default=0),
        ),
        migrations.AlterField(
            model_name='xtrace',
            name='real_log',
            field=models.IntegerField(),
        ),
    ]
