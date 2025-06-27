# parking_system/urls.py
from django.urls import path
from . import views
from django.contrib.auth import views as auth_views
from django.contrib.admin.views.decorators import staff_member_required

urlpatterns = [
    path('', views.index, name='index'),
    path('login/', views.login_view, name='login'),
    path('manage/', staff_member_required(views.manage_view), name='manage'),
    path('get_records/', views.get_records, name='get_records'),
    path('start_camera/<str:mode>/', views.start_camera, name='start_camera'),
    path('stop_camera/<str:mode>/', views.stop_camera, name='stop_camera'),
    path('get_camera_frame/<str:mode>/', views.get_camera_frame, name='get_camera_frame'),
    path('get_record/<int:record_id>/', views.get_record, name='get_record'),
    path('update_record/', views.update_record, name='update_record'),
    path('delete_record/<int:record_id>/', views.delete_record, name='delete_record'),
    path('get_stats/', views.get_stats, name='get_stats'),
    path('logout/', views.logout_view, name='logout'),
    path('register/', views.register_view, name='register'),
    path('customer/', views.customer_view, name='customer'),
    path('customer/login/', views.customer_login, name='customer_login'),
    path('admin/login/', views.admin_login, name='admin_login'),
    path('logs/', views.view_logs, name='view_logs'),
    path('logs/clear/', views.clear_logs, name='clear_logs'),

]
