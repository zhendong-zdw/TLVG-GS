import torch

class CUDATimer:
    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self._active = False

    def __enter__(self):
        if self._active:
            raise RuntimeError("Timer is already running")
            
        self.start_event.record()
        self._active = True
        return self

    def __exit__(self, *exc):
        if not self._active:
            return
        
        # Don't access CUDA if there was an exception (CUDA context may be corrupted)
        if exc[0] is not None:
            self._active = False
            return
            
        try:
            self.end_event.record()
            self._active = False
            torch.cuda.current_stream().synchronize()
        except Exception:
            # If CUDA is corrupted, just mark as inactive
            self._active = False

    @property
    def elapsed_ms(self) -> float:
        return self.start_event.elapsed_time(self.end_event)