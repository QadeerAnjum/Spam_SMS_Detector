from django.shortcuts import render
from django.http import HttpResponse
import os
import joblib 

model = joblib.load(os.path.dirname(__file__)+ "\\myModel.pkl")

# Create your views here.
def index(request):
    return render(request,"index.html")

def checkSpam(request):
    if request.method == "POST":
      message= request.POST.get("sms")
      
      finalAns = model.predict([message])[0]
      param = {"answer": finalAns}



      return render(request,"output.html",param)
    else:
      return render(request,"index.html")
