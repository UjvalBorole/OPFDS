from django.contrib import admin
from .models import *
# Register your models here.

@admin.register(History)
class AdminHistory(admin.ModelAdmin):
    list_display = ['id','user','lr','rf','xgb','mdsc','status']

@admin.register(Path)
class AdminPath(admin.ModelAdmin):
    list_display = ['id','user','path']