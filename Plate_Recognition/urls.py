from django.urls import path

from .views import (live_camera, load_params, result, stop_camera,
                    check_result, stop_camera, loading, car_entry, car_exit,
                    toggle_camera_mode, import_plates_to_db)

urlpatterns = [
    path('load_params/', load_params, name='load_params'),
    path('live_camera/', live_camera, name='live_camera'),
    path('stop_camera/', stop_camera, name='stop_camera'),
    path('result/', result, name='result'),
    path('check_result/', check_result, name='check_result'),
    path('stop_camera/', stop_camera, name='stop_camera'),
    path('loading/', loading, name='loading'),
    path('car_entry/', car_entry, name='car_entry'),
    path('car_exit/', car_exit, name='car_exit'),
    path('toggle_camera_mode/', toggle_camera_mode, name='toggle_camera_mode'),
    path('import_plates/', import_plates_to_db, name='import_plates'),
]