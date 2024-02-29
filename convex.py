from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from collections import OrderedDict
from collections import defaultdict
import matplotlib.pyplot as plt
from typing import Optional
import torch.optim as optim
from loguru import logger
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch
import h5py

# -----------------------------------
# ----------- TOY DATASET -----------
# -----------------------------------

def load(filename):
    with h5py.File(filename, 'r') as hf:
        samples = torch.from_numpy(hf['samples'][:]).to(torch.float32)
        labels = torch.from_numpy(hf['labels'][:]).to(torch.float32).view(-1, 1)
    return samples, labels

def plot(samples, labels, k):
    if samples.shape[1] < 2: raise ValueError()
    if samples.shape[1] > 2:
        pca = PCA(n_components=2)
        samples = pca.fit_transform(samples)
    for i in range(k):
        plt.scatter(samples[labels == i, 0], samples[labels == i, 1], label=f'Class {i}', alpha=0.6, s=10)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()

def plot_contour(model, samples, labels, opt):
    # mesh grid feature space
    x_min, x_max = -10, 10
    y_min, y_max = -10, 10
    h = .02  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    X_grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        predictions = model(X_grid)
        predictions = torch.argmax(predictions, dim=1)
    
    predictions = predictions.round()
    predictions = predictions.numpy()
    predictions = predictions.reshape(xx.shape)

    plt.contourf(xx, yy, predictions, alpha=0.6, cmap='Paired')
    plt.scatter(samples[:, 0], samples[:, 1], c=labels, cmap='Paired', s=10)
    plt.title(f'Contour plot of the decision boundary {opt}')
    plt.show()

# ------------------------------------------
# ----------- MODEL ARCHITECTURE -----------
# ------------------------------------------

class LinearBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        act_func: Optional[nn.Module] = None,
        bias: bool = True
    ):
        super(LinearBlock, self).__init__()
        modules = OrderedDict()
        modules["linear"] = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        modules["act_func"] = act_func if act_func else nn.Identity()
        self.block = nn.Sequential(modules)

    def forward(self, x):
        return self.block(x)

class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: list[int],
        out_features: int,
        act_func: Optional[nn.Module] = None,
        bias: bool = True
    ) -> None:
        super(MLP, self).__init__()
        assert len(hidden_features) >= 1

        layers = OrderedDict()
        hidden_dims = [in_features] + hidden_features

        for i in range(len(hidden_dims) - 1):
            layers[f"block-{i}"] = LinearBlock(
                in_features=hidden_dims[i],
                out_features=hidden_dims[i + 1],
                act_func=act_func,
                bias=bias
            )

        layers["fc"] = nn.Linear(
            in_features=hidden_dims[-1],
            out_features=out_features,
            bias=bias,
        )

        self.layers = nn.Sequential(layers)
        self.penultimate = None

    def forward(self, x):
        self.penultimate = self.layers[:-1](x)
        logits = self.layers[-1](self.penultimate)
        return logits

# ------------------------------------------
# ----------- FUNCTIONAL PYTORCH -----------
# ------------------------------------------
    
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
    
# ------------------------------------------
# ----------- SHARPNESS CALLBACK -----------
# ------------------------------------------

def metrics_in_batch(inputs, targets, names, primals, tangents, func_fwd):
    state = {"names": names, "inputs": inputs, "targets": targets}
    _, jvp = torch.autograd.functional.jvp(func_fwd, primals, tangents, state=state)
    _, hvp = torch.autograd.functional.hvp(func_fwd, primals, tangents, state=state)
    vhv = sum((v * hv).sum() for v, hv in zip(tangents, hvp))
    jvp, vhv = jvp.item(), vhv.item()
    return jvp, vhv

def metrics_in_landscape(dataloader, names, primals, tangents, func_fwd):
    agg_jvp, agg_vhv, count = 0., 0., 0.

    for inputs, targets in dataloader:
        sz = inputs.size(0)
        targets = targets.squeeze(1).long()
        state = {"names": names, "inputs": inputs, "targets": targets}
        _, jvp_ = torch.autograd.functional.jvp(func_fwd, primals, tangents, state=state)
        _, hvp_ = torch.autograd.functional.hvp(func_fwd, primals, tangents, state=state)
        vhv_ = sum((v * hv).sum() for v, hv in zip(tangents, hvp_))
        agg_jvp += jvp_.item() * sz
        agg_vhv += vhv_.item() * sz
        count += sz

    jvp = agg_jvp / count
    vhv = agg_vhv / count

    return jvp, vhv

def sharpness_callback(model, crit, dataloader, inputs, targets, prev_params, tangents, landscape='full'):
    def func_fwd(*model_params, **kwargs):
        # state-less functional forward pass
        names = kwargs.get("state").get("names")
        inputs = kwargs.get("state").get("inputs")
        targets = kwargs.get("state").get("targets")
        for name, p in zip(names, model_params):
            set_attr(model, name.split("."), p)
        return crit(model(inputs), targets)

    # make model functional for jvp and hvp
    tmp_params, names = make_functional(model)

    if landscape == "batch":
        jvp, vhv = metrics_in_batch(inputs, targets, names, prev_params, tangents, func_fwd)

    elif landscape == "full":
        jvp, vhv = metrics_in_landscape(dataloader, names, prev_params, tangents, func_fwd)
    
    else:
        raise ValueError()

    # restore the model to original state :)
    restore_functional(model, tmp_params, names)

    return jvp, vhv

# ------------------------------------------
# ----------- TRAINING FUNCTIONS -----------
# ------------------------------------------

def clone(dataloader):
    new = torch.utils.data.DataLoader(
        dataset=dataloader.dataset,
        batch_size=dataloader.batch_size
    )
    return new
    
def compute_accuracy(outputs, targets):
    _, predicted = torch.max(outputs.data, 1)
    total = targets.size(0)
    correct = (predicted == targets).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def test(model, dataloader, criterion):
    model.eval()
    with torch.no_grad():
        total_loss, total_samples, correct_predictions = 0., 0., 0.
        for _, (inputs, targets) in enumerate(dataloader):
            outputs = model(inputs)
            targets = targets.squeeze(1).long()
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            _, predicted = outputs.max(1)
            correct_predictions += predicted.eq(targets).sum().item()
    model.train()

    avg_loss = total_loss / total_samples
    avg_accuracy = 100 * correct_predictions / total_samples
    return avg_loss, avg_accuracy

def train(
    n_epochs: int,
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    opt: optim.Optimizer,
    sharpness_freq: int
):
    # initialize counters
    iters_per_epoch = len(dataloader)
    n_iters_tot = iters_per_epoch * n_epochs
    pbar = tqdm(range(n_iters_tot))
    metrics = defaultdict(list)
    n_iters = 0

    model.train()
    for epoch_idx in range(n_epochs):
        epoch_loss, epoch_acc = 0., 0.
        prev_test_loss = None

        for _, (inputs, targets) in enumerate(dataloader):
            # forward pass
            outputs = model(inputs)
            
            prev_params = tuple([p.clone().detach() for p in model.parameters()])

            # backward pass
            opt.zero_grad()
            targets = targets.view(-1).long()
            loss = criterion(outputs, targets)
            loss.backward()
            opt.step()
            
            post_params = tuple([p.clone().detach() for p in model.parameters()])

            # sharpness callback!
            if sharpness_freq > 0 and n_iters % sharpness_freq == 0:
                zipper = zip(post_params, prev_params)
                delta_W = tuple([post - prev for post, prev in zipper])
                jvp, vhv = sharpness_callback(model, criterion, clone(dataloader), inputs, targets, prev_params, delta_W, landscape='full')
                test_loss, _ = test(model, clone(dataloader), criterion)
                
                if prev_test_loss:
                    delta_loss = test_loss - prev_test_loss
                    metrics["delta_loss"].append(delta_loss)
                    metrics["jvp"].append(jvp)
                    metrics["vhv"].append(vhv)
                
                prev_test_loss = test_loss
            
            # compute and log metrics
            epoch_loss += loss.item()
            epoch_acc += compute_accuracy(outputs, targets)
            metrics["train_loss_iter"].append(loss.item())

            # move progress bar
            pbar.set_description(f"epoch {epoch_idx+1}/{n_epochs} iter {n_iters+1}/{n_iters_tot} | train loss {loss.item():.3f}")
            pbar.update(1)
            n_iters += 1

        # log metrics
        metrics["train_loss_epoch"].append(epoch_loss / iters_per_epoch)
        metrics["train_acc_epoch"].append(epoch_acc / iters_per_epoch)
    
    pbar.close()
    return metrics

# ------------------------------------------
# ----------- TRAINING PIPELINE -----------
# ------------------------------------------

def set_seed(seed=123):
    if seed is not None: 
        torch.manual_seed(seed)

def pipeline(bs, n_epochs, sharpness_freq, opt):
    logger.info(f"Starting training for {opt} on {n_epochs} epochs")

    set_seed()
    model = MLP(
        in_features=2, 
        hidden_features=[64, 32, 16, 8], 
        out_features=5, 
        act_func=nn.ReLU()
    )

    if opt == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=1e-2, weight_decay=1e-4)
    elif opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss()

    samples, labels = load('toy/clouds-n1000-k5-d2-gaussian.h5')
    dataset = TensorDataset(samples, labels)
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)

    plot_contour(model, samples, labels, opt)
    metrics = train(n_epochs, model, dataloader, criterion, optimizer, sharpness_freq)
    plot_contour(model, samples, labels, opt)

    return metrics

# ------------------------------------------
# ----------------- MAIN -------------------
# ------------------------------------------

metrics_sgd = pipeline(bs=10, n_epochs=10, sharpness_freq=10, opt="sgd")
metrics_adam = pipeline(bs=10, n_epochs=10, sharpness_freq=10, opt="adam")

if 'vhv' in metrics_sgd:
    print("SGD vhv mean", np.mean(metrics_sgd["vhv"]))
    print("Adam vhv mean", np.mean(metrics_adam["vhv"]))
