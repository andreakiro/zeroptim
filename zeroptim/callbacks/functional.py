# https://discuss.pytorch.org/t/combining-functional-jvp-with-a-nn-module/81215/2

def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])

def set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)

def make_functional(model):
    # Remove all the parameters in the model
    orig_params = tuple(model.parameters())
    names = []
    for name, p in list(model.named_parameters()):
        del_attr(model, name.split("."))
        names.append(name)
    return orig_params, names

def restore_functional(model, orig_params, names):
    # Restore all the parameters in the model
    for name, p in zip(names, orig_params):
        set_attr(model, name.split("."), p)
