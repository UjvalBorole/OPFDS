from django.urls import path
from app import views
from django.contrib.auth import views as auth_view
from .forms import LoginForm,passwordchange,setpasswordconfirm
from .forms import PasswordReset


urlpatterns = [
    path('home/', views.home,name="home"),
    path('df/', views.dataframe,name = "df"),
    path('ht/', views.histogram,name="ht"),
    path('knn/', views.Knn,name = "knn"),
    path('rf/', views.randomForest,name = "rf"),
    path('pd/', views.predict,name = "pd"),
    path('difal/', views.diffalgoAccu,name = "difal"),
    path('hist/', views.history,name = "hist"),
    path('clrhist/', views.clrhistory,name = "clrhist"),
    path('about/', views.about,name = "about"),
    path('profile/', views.profile,name = "profile"),
    path('reset/', views.reset,name = "reset"),
    # path('', views.train),


    path('changepassword/', auth_view.PasswordChangeView.as_view(template_name='auth/changepassword.html',form_class=passwordchange,success_url='/changepassworddone/'), name='changepassword'),
    path('changepassworddone/', auth_view.PasswordChangeDoneView.as_view(template_name='auth/changepassworddone.html'), name='changepassworddone'),
    # path('changepass/',views.changepass,name='changepassword'),

    path('password-reset/', auth_view.PasswordResetView.as_view(template_name='auth/password_reset.html',form_class=PasswordReset),name='passwordreset'),
    path('password-reset/done/', auth_view.PasswordResetDoneView.as_view(template_name='auth/password_reset_done.html'),name='password_reset_done'),
    path('password-reset-confirm/<uidb64>/<token>/', auth_view.PasswordResetConfirmView.as_view(template_name='auth/password_reset_confirm.html',form_class=setpasswordconfirm),name='password_reset_confirm'),
    path('password-reset-complete/', auth_view.PasswordResetCompleteView.as_view(template_name='auth/password_reset_complete.html'),name='password_reset_complete'),

    
    path('accounts/login/', auth_view.LoginView.as_view(template_name='auth/login.html',authentication_form=LoginForm), name='login'),
    path('logout/',views.custom_logout ,name='logout'),

    path('registration/', views.customerregistration.as_view(), name='customerregistration'),
]
