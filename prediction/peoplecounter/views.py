from django.shortcuts import render

from django.shortcuts import render, redirect
from django.urls import reverse

from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from peoplecounter.forms import LoginForm, RegisterForm

from django.utils import timezone
from datetime import datetime

# Create your views here.

def _andrew_check(action_function):
    def my_wrapper_function(request, *args, **kwargs):
        if request.user.is_authenticated and not request.user.email.endswith("@andrew.cmu.edu"):
            return render(request, '../templates/unauthorized.html')
        return action_function(request, *args, **kwargs)

    return my_wrapper_function

@_andrew_check
@login_required
def global_stream_action(request):
    # context = {}
    # if request.method == 'GET':
    #     context['form'] = RegisterForm()
    #     return render(request, '../templates/register.html', context)

    context = _compute_context_global()

    return render(request, '../templates/global.html', context)

def _compute_context_global():
    curYear = datetime.now().year
    curMonth = datetime.now().month
    curDay = datetime.now().day
    curHour = datetime.now().strftime('%H')
    curMin = datetime.now().strftime('%M')

    context = {"curYear": curYear,
               "curMonth": curMonth,
               "curDay": curDay,
               "curHour": curHour,
               "curMin": curMin,
    }
    return context

def login_action(request):
    context = {}

    # Just display the registration form if this is a GET request.
    if request.method == 'GET':
        context['form'] = LoginForm()
        return render(request, '../templates/login.html', context)

    # Creates a bound form from the request POST parameters and makes the 
    # form available in the request context dictionary.
    form = LoginForm(request.POST)
    context['form'] = form

    # Validates the form.
    if not form.is_valid():
        return render(request, '../templates/login.html', context)

    new_user = authenticate(username=form.cleaned_data['username'],
                            password=form.cleaned_data['password'])

    login(request, new_user)
    return redirect(reverse('home'))


def logout_action(request):
    logout(request)
    return redirect(reverse('login'))


def register_action(request):
    context = {}

    # Just display the registration form if this is a GET request.
    if request.method == 'GET':
        context['form'] = RegisterForm()
        return render(request, '../templates/register.html', context)

    # Creates a bound form from the request POST parameters and makes the 
    # form available in the request context dictionary.
    form = RegisterForm(request.POST)
    context['form'] = form

    # Validates the form.
    if not form.is_valid():
        return render(request, '../templates/register.html', context)

    # At this point, the form data is valid.  Register and login the user.
    new_user = User.objects.create_user(username=form.cleaned_data['username'], 
                                        password=form.cleaned_data['password'],
                                        email=form.cleaned_data['email'],
                                        first_name=form.cleaned_data['first_name'],
                                        last_name=form.cleaned_data['last_name'])
    new_user.save()

    new_user = authenticate(username=form.cleaned_data['username'],
                            password=form.cleaned_data['password'])

    login(request, new_user)
    return redirect(reverse('home'))