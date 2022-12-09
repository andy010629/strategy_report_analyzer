
def remove_timezone(dt):
    return dt.replace(tzinfo=None)
