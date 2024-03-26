import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import make 
import subprocess


class Watcher:
    DIRECTORY_TO_WATCH = "/Users/antoinemagron/Documents/EPFL/PDM/crchum/"

    def __init__(self):
        self.observer = Observer()

    def run(self):
        event_handler = Handler()
        self.observer.schedule(event_handler, self.DIRECTORY_TO_WATCH, recursive=True)
        self.observer.start()
        try:
            while True:
                time.sleep(10)
        except:
            self.observer.stop()
            print("Error")
        
        self.observer.join()


class Handler(FileSystemEventHandler):

    @staticmethod
    def on_any_event(event):
        if event.is_directory:
            return None

        elif event.event_type in ['modified', 'created']:
            if("__pycache__" not in event.src_path 
               and"__init__.py" not in event.src_path 
               and ".py" in event.src_path):
                
                print(f"Received {event.event_type} event - %s." % event.src_path)
                print("> Rebuild :")
                subprocess.call("./dependencies.sh")



if __name__ == '__main__':
    w = Watcher()
    w.run()
