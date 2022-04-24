from http.client import HTTPResponse
from multiprocessing import context
from django.shortcuts import render
from django.http import HttpResponse
from majorproject import final
# Create your views here.
from joblib import load
model=load('./models/model_5000.joblib')


def index(request):
    context={'a':1}
    return render(request,'mp/index.html',context)

def predict(request):
    context={'output':None}
    if request.method=='POST':
        temp = dict()
        temp['title'] = request.POST.get('title')
        temp['tags'] = request.POST.get('tags')
        temp['body'] = request.POST.get('body')
        temp['code'] = request.POST.get('code')
        var=final.pred(temp)
        output=model.predict(list(var['document']))
        print(output)
        context['output']=output[0]
        return render(request,'mp/predict.html',context)
    
    return HttpResponse('This page does not exists')