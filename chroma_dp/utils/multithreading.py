import queue
import threading
from typing import Callable, Any, Optional

MAX_QUEUE_SIZE = 20  # Adjust as needed
data_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)


class ThreadConsumer(threading.Thread):
    def __init__(self, data_queue: queue.Queue, callback: Callable[[Any], None], timeout: Optional[int] = None,
                 stop_event: Optional[threading.Event] = None):
        super().__init__()
        self.data_queue = data_queue
        self.callback = callback
        if not stop_event and not timeout:
            raise ValueError("Either timeout or stop_event must be set")
        self.timeout = timeout
        self.stop_event = stop_event

    def run(self):
        while True:
            try:
                data = self.data_queue.get(timeout=self.timeout)
                if self.stop_event and self.stop_event.is_set():
                    break
                if data is None:
                    break
                # TODO do we need error handling or just let the thread die?
                self.callback(data)
            except queue.Empty:
                pass
