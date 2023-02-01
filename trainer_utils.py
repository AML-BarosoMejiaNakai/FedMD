import torch.optim as optim

class EarlyStop:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def __call__(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def load_optimizer(model, params):
    assert (model is not None and params is not None and \
        "optimizer" in params and "lr" in params)
    
    optimizer = None
    lr = params["lr"]
    weight_decay = params["weight_decay"] if "weight_decay" in params else 0
    if params["optimizer"] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif params["optimizer"] == "SGD":
        momentum = params["momentum"] if "momentum" in params else 0
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    
    return optimizer