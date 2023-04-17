"""
Within-domain transfer learning
"""
import torch
from dataloader import train_loader, val_loader, test_loader
from loss import loss_fn
from config import config
from tqdm import tqdm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    val_acc, val_loss = validate_model(val_loader)
    print("Val Acc {:.04f}%\t Val Loss {:.04f}".format(val_acc, val_loss))

if __name__ == '__main__':
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_mobilenetv2_x1_0", pretrained=True)
    model.classifier = torch.nn.Flatten()
    model.to(DEVICE)
    main()
