from django.db import models
from django.contrib.auth.models import User

class History(models.Model):
    user = models.ForeignKey(User,on_delete=models.CASCADE)
    lr = models.FloatField()
    rf = models.FloatField()
    xgb = models.FloatField()
    mdsc = models.FloatField()
    status = models.CharField(max_length=500)

class Path(models.Model):
    user = models.ForeignKey(User,on_delete=models.CASCADE)
    path = models.CharField(max_length=2000)




