from django.db import models

# Create your models here.

from django.forms import ModelForm

class UploadDigit(models.Model):
    pic = models.FileField(upload_to="images/")    
    upload_date=models.DateTimeField(auto_now_add =True)

# FileUpload form class.
class UploadDigitForm(ModelForm):
    class Meta:
        model = UploadDigit
        fields = ('pic',)
