import shutil
import os
import time
from datetime import datetime
import argparse
import numpy as np
import random

from transformers.optimization import AdamW
from transformers.optimization import get_linear_schedule_with_warmup

import torch
import torch.optim as optim
from torchsample.transforms import RandomRotate, RandomTranslate, RandomFlip, Compose
from torchvision import transforms
from tensorboardX import SummaryWriter

from sklearn import metrics

import model
from dataloader import MRDataset, stack_collate


def train_model(model, train_loader, epoch, num_epochs, optimizer,scheduler, writer, current_lr, log_every=100):
    model.train()
    
    y_preds = []
    y_trues = []
    losses = []
    training_preds = []
    training_trues = []
    
    for i, (mris_batch,images_size, labels, weights) in enumerate(train_loader):

        if torch.cuda.is_available():
            mris_batch = mris_batch.cuda()
            labels = labels.cuda()
            weights = weights.cuda()

        # Forward pass
        prediction = model.forward(mris_batch,images_size)

        loss = torch.nn.BCEWithLogitsLoss(pos_weight=weights)(prediction, labels) # 
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        loss_value = loss.item()
        losses.append(loss_value)
        # Keep data for log and final evaluation
        labels = labels.detach().cpu().numpy()
        y_trues.extend(labels)
        training_trues.extend(labels)
        # 
        probas = torch.sigmoid(prediction).detach().cpu().numpy()
        y_preds.extend(probas)
        training_preds.extend(probas)

        writer.add_scalar('Train/Loss', loss_value,
                          epoch * len(train_loader) + i)

        if (i % log_every == 0) & (i > 0):
            try:
                # Eval training set until now
                train_auc = metrics.roc_auc_score(y_trues, y_preds)
            except:
                train_auc = 0.5
            training_preds, training_trues = [], []
            writer.add_scalar('Train/auc', train_auc, epoch * len(train_loader) + i)
            
            print('''[Epoch: {0} / {1} |Single batch number : {2} / {3} ]| avg train loss {4} | train auc : {5} | lr : {6}'''.
                  format(
                      epoch + 1,
                      num_epochs,
                      i,
                      len(train_loader),
                      np.round(np.mean(losses), 4),
                      np.round(train_auc, 4),
                      get_lr(optimizer)
                  )
                  )
    auc = metrics.roc_auc_score(y_trues, y_preds)
    writer.add_scalar('Train/auc', auc, epoch + i)

    train_loss_epoch = np.round(np.mean(losses), 4)
    train_auc_epoch = np.round(auc, 4)
    return train_loss_epoch, train_auc_epoch


def evaluate_model(model, val_loader, epoch, num_epochs, writer, current_lr, log_every=2):
    model.eval()
    y_preds = []
    y_trues = []
    losses = []

    for i, (mris_batch,images_size, labels, weights) in enumerate(val_loader):

        if torch.cuda.is_available():
            mris_batch = mris_batch.cuda()
            labels = labels.cuda()
            weights = weights.cuda()
        # Forward pass
        with torch.no_grad():
            prediction = model.forward(mris_batch,images_size)

        loss = torch.nn.BCEWithLogitsLoss(pos_weight=weights)(prediction, labels)

        loss_value = loss.item()
        losses.append(loss_value)
        # Keep data for log and final evaluation
        labels = labels.detach().cpu().numpy()
        y_trues.extend(labels)
        # 
        probas = torch.sigmoid(prediction).detach().cpu().numpy()
        y_preds.extend(probas)


        writer.add_scalar('Valid/Loss', loss_value,
                          epoch * len(val_loader) + i)


        if (i % log_every == 0) & (i > 0):
            try:
                valid_auc = metrics.roc_auc_score(y_trues, y_preds)
            except:
                valid_auc = 0.5
            writer.add_scalar('Valid/auc', valid_auc, epoch * len(val_loader) + i)
            print('''[Epoch: {0} / {1} |Single batch number : {2} / {3} ]| avg valid loss {4} | valid auc : {5} | lr : {6}'''.
                  format(
                      epoch + 1,
                      num_epochs,
                      i,
                      len(val_loader),
                      np.round(np.mean(losses), 4),
                      np.round(valid_auc, 4),
                      current_lr
                  )
                  )
    auc = metrics.roc_auc_score(y_trues, y_preds)
    writer.add_scalar('Valid/auc', valid_auc, epoch * len(val_loader) + i)
    val_auc_epoch = np.round(auc, 4)
    val_loss_epoch = np.round(np.mean(losses), 4) 
    return val_loss_epoch, val_auc_epoch

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def run(args):
    base_folder = os.getenv("BASE_FOLDER",".")
    log_root_folder = f"{base_folder}/logs/{args.task}/{args.plane}/"
    if args.flush_history == 1:
        objects = os.listdir(log_root_folder)
        for f in objects:
            if os.path.isdir(log_root_folder + f):
                shutil.rmtree(log_root_folder + f)

    now = datetime.now()
    logdir = log_root_folder + now.strftime("%Y%m%d-%H%M%S") + "/"
    os.makedirs(logdir)

    writer = SummaryWriter(logdir)


    augmentor = Compose([
        transforms.Lambda(lambda x: torch.Tensor(x)),
        RandomRotate(25),
        RandomTranslate([0.11, 0.11]),
        RandomFlip(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1, 1).permute(1, 0, 2, 3)),
    ]) if args.augment else None
    batch_size = args.batch_size
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    train_dataset = MRDataset(f'{base_folder}/data/', args.task,
<<<<<<< HEAD
                              args.plane, transform=augmentor, train=True)
=======
                              args.plane, train=True,transform=augmentor)
>>>>>>> da938afd06eb1f799ce2cbc1c60bad30ab6c60a1
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=6, drop_last=False, collate_fn=stack_collate, generator=g)

    validation_dataset = MRDataset(
        f'{base_folder}/data/', args.task, args.plane, train=False)
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=-True, num_workers=6, drop_last=False,collate_fn=stack_collate, generator=g)

    mrnet = model.MRNet()

    if torch.cuda.is_available():
        mrnet = mrnet.cuda()

    # GEt optimizer
    no_decay = ["bias", "gamma", "beta"]
    parameters_without_decay = []
    parameters_with_decay = []
    for n, p in mrnet.named_parameters():
        if any(t in n for t in no_decay):
            parameters_without_decay.append(p)
        else:
            parameters_with_decay.append(p)
    optimizer_grouped_parameters = [
        {"params": parameters_with_decay, "weight_decay": 0.1, "lr": args.lr},
        {"params": parameters_without_decay, "weight_decay": 0.0, "lr": args.lr},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, correct_bias=False)
    warmup_proportion = 0.01
    len_train_data = len(train_loader)
    num_train_steps = int(len_train_data / batch_size) * args.epochs
    num_warmup_steps = int(num_train_steps * warmup_proportion)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_train_steps,
    )
    best_val_loss = float('inf')
    best_val_auc = float(0)

    num_epochs = args.epochs
    iteration_change_loss = 0
    patience = args.patience
    log_every = args.log_every

    t_start_training = time.time()

    for epoch in range(num_epochs):
        current_lr = get_lr(optimizer)

        t_start = time.time()
        
        train_loss, train_auc = train_model(
            mrnet, train_loader, epoch, num_epochs, optimizer,scheduler, writer, current_lr, log_every)
        val_loss, val_auc = evaluate_model(
            mrnet, validation_loader, epoch, num_epochs, writer, current_lr)


        t_end = time.time()
        delta = t_end - t_start

        print("train loss : {0} | train auc {1} | val loss {2} | val auc {3} | elapsed time {4} s".format(
            train_loss, train_auc, val_loss, val_auc, delta))

        iteration_change_loss += 1
        print('-' * 30)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            if bool(args.save_model):
                file_name = f'model_{args.prefix_name}_{args.task}_{args.plane}_val_auc_{val_auc:0.4f}_train_auc_{train_auc:0.4f}_epoch_{epoch+1}.pth'
                for f in os.listdir(f'{base_folder}/models/'):
                    if (args.task in f) and (args.plane in f) and (args.prefix_name in f):
                        os.remove(f'{base_folder}/models/{f}')
                torch.save(mrnet, f'{base_folder}/models/{file_name}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            iteration_change_loss = 0

        if iteration_change_loss == patience:
            print('Early stopping after {0} iterations without the decrease of the val loss'.
                  format(iteration_change_loss))
            break

    t_end_training = time.time()
    print(f'training took {t_end_training - t_start_training} s')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, 
                        choices=['abnormal', 'acl', 'meniscus'], default="acl")
    parser.add_argument('-p', '--plane', type=str,
                        choices=['sagittal', 'coronal', 'axial'], default='sagittal')
    parser.add_argument('--prefix_name', type=str, default='test_model')
    parser.add_argument('--augment', type=int, choices=[0, 1], default=1)
    parser.add_argument('--lr_scheduler', type=str,
                        default='step', choices=['plateau', 'step'])
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--flush_history', type=int, choices=[0, 1], default=0)
    parser.add_argument('--save_model', type=int, choices=[0, 1], default=1)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    run(args)
