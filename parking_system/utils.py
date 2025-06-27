from .models import SystemLog

def log_operation(request, operation_type, detail):
    """记录系统操作日志"""
    SystemLog.objects.create(
        user=request.user if request.user.is_authenticated else None,
        operation_type=operation_type,
        operation_detail=detail,
        ip_address=request.META.get('REMOTE_ADDR')
    )