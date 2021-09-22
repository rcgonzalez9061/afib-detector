from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from collections import OrderedDict

from models import MergedModel, Hsieh2020, Afib_CNN

from utils import load_project_config, load_model
PROJECT_CONFIG = load_project_config()
PROJECT_DIR = PROJECT_CONFIG['project_dir']
DATA_DIR = PROJECT_CONFIG['data_dir']

    
def epoch_train(
    model,
    device,
    dataset,
    optimizer,
    epoch,
    train_size,
    batch_size,
    log_interval=10,
    dry_run=False,
    print_progress=False
):
    model.train()

    batch_idx = 1
    num_remaining = train_size
    total_loss = 0
    correct = 0

    while num_remaining > 0:
        batch_size = min((num_remaining, batch_size))
        idx, data, target = dataset.get_batch(batch_size)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)

        loss.backward()
        optimizer.step()
        
        num_remaining -= batch_size
        batch_idx += 1
        total_loss += F.nll_loss(output, target, reduction='sum').item()
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

        
        if batch_idx % log_interval == 0 or num_remaining==0:
            if print_progress == 'pct':
                num_completed = train_size - num_remaining
                pct_complete = 100. * num_completed / train_size
                print(f'\rTraining...{pct_complete:.1f}%', end='')
#                 if num_remaining==0:
#                     print()
            elif print_progress == 'full':
                num_completed = train_size - num_remaining
                pct_complete = 100. * num_completed / train_size
                print(f'Train Epoch: {epoch} [{num_completed}/{train_size} ({pct_complete:.0f}%)]\tLoss: {loss.item():.6f}')
                if dry_run:
                    break
    avg_loss = total_loss / train_size
#     if print_progress:
#         print(f'Train Epoch: {epoch} Average Loss:{avg_loss: .6f}')

    acc = 100. * correct / train_size
    return acc, avg_loss
        
def epoch_test(
    model,
    device, 
    dataset,
    test_size,
    test_batch_size,
    print_progress=False
):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        num_remaining = test_size
        while num_remaining > 0:
            batch_size = min((num_remaining, test_batch_size))
            idx, data, target = dataset.get_batch(batch_size)
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            num_remaining -= batch_size

    avg_loss = test_loss / test_size
    accuracy = 100. * correct / test_size
    if print_progress:
        print(f'Test set: Average loss: {avg_loss:.4f}, Accuracy: {correct}/{test_size} ({accuracy:.1f}%)')
    return accuracy, avg_loss

def save_model(model, path):
    torch.save(model.state_dict(), path)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='AFib Detector CNN')
    parser.add_argument('--data-path', action='store_true', default=DATA_DIR,
                        help='Location of physionet data')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--train-size', type=int, default=500, metavar='N_tr',
                        help='Size of training set (default: 500)')
    parser.add_argument('--test-size', type=int, default=1000, metavar='N_ts',
                        help='Size of test set (default: 1000)')
    parser.add_argument('--window-size', type=int, default=2500, metavar='N',
                        help='Size of window (default: 2500)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--print-progress', action='store_true', default=False,
                        help='Output while training')
    args = parser.parse_args()
    args_dict = vars(args)
    model_train_kwargs = {
        'train_size': args.train_size,
        'batch_size': args.batch_size,
        'log_interval': args.log_interval,
        'dry_run': args.dry_run
    }
    model_test_kwargs = {
        'test_size': args.test_size,
        'test_batch_size': args.test_batch_size
    }
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    
    print('Loading data...')
    # Nonfunction, need to redo
#     train_dataset, test_dataset = load_train_test_datasets(args.data_path, args.window_size)

#     model = Afib_CNN(
#         args.window_size, 
#         2, 
#         conv_size=87, 
#         max_pool_size=23, 
#         conv1_out_channels=28, 
#         conv2_out_channels=59
#     )
#     model = model.to(device)
#     optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
#     scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
#     
#     print('Training model...')
#     for epoch in range(1, args.epochs + 1):
#         model.train_model(device, train_dataset, optimizer, epoch, print_progress=args.print_progress, **model_train_kwargs)
#         accuracy = model.test_model(device, test_dataset, print_progress=True, **model_test_kwargs)
#         scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "afib_detector_cnn.pt")
        
if __name__ == '__main__':
    main()