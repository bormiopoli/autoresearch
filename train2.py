def get_params() -> dict[str, Any]:
    pass

class TimeBudgetCallback(keras.callbacks.Callback):
    """
    Stop training once the fixed wall-clock budget is reached.
    
    This is preferable to simply comparing epoch counts if experiments
    have different computational costs.
    """
    
    def __init__(self, seconds: float):
        self.seconds = seconds
    
    def on_train_begin(self, logs=None):
        pass
    
    def on_epoch_end(self, epoch, logs=None):
        pass

def train():
    pass
