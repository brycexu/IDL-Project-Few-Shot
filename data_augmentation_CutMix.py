"""
Train a Protypical Network with CutMix Data Augmentation
Reference : https://github.com/clovaai/CutMix-PyTorch
"""
import torch
from dataloader import train_loader, val_loader, test_loader
from model import ResNet12
from loss import loss_fn
from config import config
import gc
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

from tqdm import tqdm

from torch.autograd import Variable

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def train_model(train_loader):
    model.train()
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train')
    train_loss = []
    train_acc = []
    for batch_index, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)
        r = np.random.rand(1)
        if r < config['cutmix_prob']:
            lam = np.random.beta(config['beta'], config['beta'])
            rand_index = torch.randperm(inputs.size()[0]).cuda()
            target_a = targets
            target_b = targets[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
            inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
            # compute output
            output = model(inputs)
            loss = loss_fn(output, target_a,config["n_shot"])[0]* lam + loss_fn(output, target_b,config["n_shot"])[0]* (1. - lam)
            acc = loss_fn(output, target_a,config["n_shot"])[1]* lam + loss_fn(output, target_b,config["n_shot"])[1]* (1. - lam)
        else:
            output = model(inputs)
            loss,acc = loss_fn(output, targets,config["n_shot"])

        loss = loss.to(DEVICE)
        train_loss.append(loss.item())
        train_acc.append(acc.item())
        batch_bar.set_postfix(
            acc="{:.04f}%".format(100 * sum(train_acc) / len(train_acc)),
            loss="{:.04f}".format(sum(train_loss) / len(train_loss)),
            lr="{:.04f}".format(float(optimizer.param_groups[0]['lr']))
        )
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_bar.update()
    batch_bar.close()
    return 100 * sum(train_acc)/len(train_acc), sum(train_loss)/len(train_loss)

def validate_model(val_loader):
    model.eval()
    batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, position=0, leave=False, desc='Val')
    val_loss = []
    val_acc = []
    for batch_index, (inputs, targets) in enumerate(val_loader):
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)
        output = model(inputs)
        loss, acc = loss_fn(input=output, target=targets, n_support=config["n_shot"])
        loss = loss.to(DEVICE)
        val_loss.append(loss.item())
        val_acc.append(acc.item())
        batch_bar.set_postfix(
            acc="{:.04f}%".format(100 * sum(val_acc) / len(val_acc)),
            loss="{:.04f}".format(sum(val_loss) / len(val_loss))
        )
        batch_bar.update()
    batch_bar.close()
    return 100 * sum(val_acc)/len(val_acc), sum(val_loss)/len(val_loss)

def main():
    torch.cuda.empty_cache()
    gc.collect()
    best_valacc = 0.0
    best_lev_dist = float("inf")
    for epoch in range(0, config['epochs']):
        curr_lr = float(optimizer.param_groups[0]['lr'])
        train_acc, train_loss = train_model(train_loader)
        print("\nEpoch {}/{}: \nTrain Acc {:.04f}%\t Train Loss {:.04f}\t Learning Rate {:.04f}".format(
            epoch + 1,
            config['epochs'],
            train_acc,
            train_loss,
            curr_lr))
        scheduler.step()
        val_acc, val_loss = validate_model(val_loader)
        print("Val Acc {:.04f}%\t Val Loss {:.04f}".format(val_acc, val_loss))
        if val_acc >= best_valacc:
            print("Saving model")
            torch.save(model.state_dict(), "./checkpoint.pth")
            best_valacc = val_acc

if __name__ == '__main__':
    model = ResNet12().to(DEVICE)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, gamma=0.5, step_size=20)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=0)
    scaler = torch.cuda.amp.GradScaler()
    main()