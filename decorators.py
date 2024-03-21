def logging(func):
    def wrapper(*args, **kwargs):
        print("Before calling the function")
        result = func(*args, **kwargs)
        print("After calling the function")
        return result
    return wrapper

## Decorator to manage dependencies
def expose(f):
    f.exposed = True
    return f