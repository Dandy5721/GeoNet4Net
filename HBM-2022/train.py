import torch
import os
from torch.utils.data import DataLoader
from torch.optim import SGD
from tqdm import tqdm
from torchinfo import summary
import numpy as np
import pandas as pd

from dataset import RtFCDataset
from spdnet.optimizer import StiefelMetaOptimizer
from net import MSNet
from mean_shift import distance_loss, SPDSpectralClustering
from utils import cluster_score, plot_epochs, AverageMeter
import warnings

warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if __name__ == '__main__':
    num_classes = 3
    windows = [30]
    num_workers = 8
    scan = None
    seed = 0

    resume = None
    resume_log = None
    transpose = False

    use_cuda = False
    save_result = True
    save_test_result = True
    total_epochs = 5
    log_columns = [
        'epoch',
        'train_loss',
        'train_acc',
        'test_loss',
        'test_acc',
        'val_loss',
        'val_acc',
    ]

    device = torch.device('cuda:0' if use_cuda else 'cpu')

    input_dir = 'data/demo/bolds'
    label_dir = 'data/demo/labels/label.csv'
    output_dir_name = 'demo'

    for window in windows:
        torch.manual_seed(seed)
        np.random.seed(seed)

        print('\n############### current window = %d ###############\n' % window)

        if scan:
            train_result_path = os.path.join(
                'train_results', output_dir_name, '{}/window={}'.format(scan, window)
            )
            test_result_path = os.path.join(
                'test_results', output_dir_name, '{}/window={}'.format(scan, window)
            )
        else:
            train_result_path = os.path.join(
                'train_results', output_dir_name, 'window={}'.format(window)
            )
            test_result_path = os.path.join(
                'test_results', output_dir_name, 'window={}'.format(window)
            )

        models_save_path = os.path.join(train_result_path, 'models_save')

        num_sample = len(
            [path for path in os.listdir(input_dir) if not path.startswith('.')]
        )
        num_train = num_sample // 10 * 6
        num_val = (num_sample - num_train) // 2
        num_test = num_sample - num_train - num_val

        print(f'num_sample = {num_sample}')
        print(f'num_train = {num_train}')
        print(f'num_val = {num_val}')
        print(f'num_test = {num_test}')

        train_dataset = RtFCDataset(
            input_dir,
            label_dir,
            slice=slice(num_train),
            window=window,
            normalize=1e-3,
            transpose=transpose,
            delimiter=None,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=1, shuffle=True, num_workers=num_workers
        )

        val_dataset = RtFCDataset(
            input_dir,
            label_dir,
            slice=slice(num_train, num_train + num_val),
            window=window,
            normalize=1e-3,
            transpose=transpose,
            delimiter=None,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=False, num_workers=num_workers
        )

        test_dataset = RtFCDataset(
            input_dir,
            label_dir,
            slice=slice(-num_test, None),
            window=window,
            normalize=1e-3,
            transpose=transpose,
            delimiter=None,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=num_workers
        )

        model = MSNet(num_classes=num_classes)
        model.to(device)

        print(train_result_path)
        if save_test_result:
            os.makedirs(test_result_path, exist_ok=True)

        optimizer = SGD(model.parameters(), lr=0.005, weight_decay=1e-5, momentum=0.9)

        if resume:
            print('resume: ' + resume)
            checkpoint = torch.load(resume)
            start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['opt'])
            model.load_state_dict(checkpoint['model'])

        os.makedirs(models_save_path, exist_ok=True)

        if save_result:
            pd.DataFrame(columns=log_columns).to_csv(
                os.path.join(train_result_path, 'log.csv'), mode='a', index=False
            )
            with open(os.path.join(train_result_path, 'net.txt'), 'w') as f:
                size = train_dataset[0][0].shape[-1]
                a = torch.randn(8, size, size)
                a = a @ a.transpose(-2, -1) + 1e-4 * torch.eye(size)
                model_str = str(summary(model, input_data=a, device=device, depth=4))
                f.write(model_str)

        criterion = distance_loss

        # optimizer = torch.optim.Adadelta(model.parameters())
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        optimizer = StiefelMetaOptimizer(optimizer)

        def train(data_loader):
            model.train()
            losses = AverageMeter()
            purities = AverageMeter()
            nmis = AverageMeter()
            bar = tqdm(enumerate(data_loader), total=len(data_loader))
            for batch_idx, (inputs, targets) in bar:
                inputs = inputs.squeeze().to(device)
                targets = targets.squeeze().to(device)

                optimizer.zero_grad()
                outputs = model(inputs)

                loss = criterion(outputs, targets)

                loss.backward()
                optimizer.step()

                clustering = SPDSpectralClustering(n_clusters=num_classes, bandwidth=50)
                labels_pred = clustering.fit(outputs).labels_

                purity, _, nmi = cluster_score(targets.cpu(), labels_pred)

                losses.update(loss.item())
                purities.update(purity)
                nmis.update(nmi)

                bar.set_description(
                    f'Loss: {losses.avg:.4f} | purity: {purities.avg:.4f} | nmi: {nmis.avg:.4f}'
                )
            return losses.avg, purities.avg

        @torch.no_grad()
        def test(data_loader):
            model.eval()
            losses = AverageMeter()
            purities = AverageMeter()
            nmis = AverageMeter()
            bar = tqdm(enumerate(data_loader), total=len(data_loader))
            all_purity = []
            labels = []
            for batch_idx, (inputs, targets) in bar:
                inputs = inputs.squeeze().to(device)
                targets = targets.squeeze().to(device)

                outputs = model(inputs)

                loss = criterion(outputs, targets)

                clustering = SPDSpectralClustering(n_clusters=num_classes, bandwidth=50)
                labels_pred = clustering.fit(outputs).labels_

                purity, _, nmi = cluster_score(targets.cpu(), labels_pred)

                losses.update(loss.item())
                purities.update(purity)
                nmis.update(nmi)
                labels.append(labels_pred)
                all_purity.append(purity)

                bar.set_description(
                    f'Loss: {losses.avg:.4f} | purity: {purities.avg:.4f} | nmi: {nmis.avg:.4f}'
                )

            return losses.avg, purities.avg, labels, all_purity

        train_losses = []
        train_accs = []

        test_losses = []
        test_accs = []

        val_losses = []
        val_accs = []

        epochs = []
        best_acc = 0

        start_epoch = 1

        if resume_log:
            print('resume_log: ' + resume_log)
            log = pd.read_csv(resume_log)
            train_losses, train_accs = (
                log['train_loss'].to_list(),
                log['train_acc'].to_list(),
            )
            test_losses, test_accs = (
                log['test_loss'].to_list(),
                log['test_acc'].to_list(),
            )
            val_losses, val_accs = log['val_loss'].to_list(), log['val_acc'].to_list()
            epochs = log['epoch'].to_list()
            best_acc = max(test_accs)
            start_epoch = max(epochs) + 1

            for i, epoch in enumerate(epochs):
                print('\nEpoch: %d' % epoch)
                print('Loss: %.4f | purity: %.4f' % (train_losses[i], train_accs[i]))
                print('Loss: %.4f | purity: %.4f' % (test_losses[i], test_accs[i]))
                print('Loss: %.4f | purity: %.4f' % (val_losses[i], val_accs[i]))

        for epoch in range(start_epoch, start_epoch + total_epochs):
            print('\nEpoch: %d' % epoch)
            train_loss, train_acc = train(train_loader)
            test_loss, test_acc, labels, accs = test(test_loader)
            val_loss, val_acc, _, _ = test(val_loader)

            if val_acc > best_acc:
                print('best')
                best_acc = test_acc
                if save_test_result:
                    for i in range(len(labels)):
                        np.savetxt(
                            os.path.join(test_result_path, '%d.csv') % (i + 1),
                            labels[i],
                            fmt='%d',
                            delimiter=',',
                        )
                    np.savetxt(
                        os.path.join(test_result_path, 'ans.csv'),
                        [accs],
                        fmt='%f',
                        delimiter=',',
                    )

            epochs.append(epoch)

            train_losses.append(train_loss)
            train_accs.append(train_acc)

            test_losses.append(test_loss)
            test_accs.append(test_acc)

            val_losses.append(val_loss)
            val_accs.append(val_acc)

            if save_result:
                plot_epochs(
                    os.path.join(train_result_path, 'loss.svg'),
                    [train_losses, test_losses, val_losses],
                    epochs,
                    xlabel='epoch',
                    ylabel='loss',
                    legends=['train', 'test', 'val'],
                    max=False,
                )
                plot_epochs(
                    os.path.join(train_result_path, 'acc.svg'),
                    [train_accs, test_accs, val_accs],
                    epochs,
                    xlabel='epoch',
                    ylabel='accuracy',
                    legends=['train', 'test', 'val'],
                )

                pd.DataFrame(
                    [
                        [
                            epoch,
                            train_loss,
                            train_acc,
                            test_loss,
                            test_acc,
                            val_loss,
                            val_acc,
                        ]
                    ]
                ).to_csv(
                    os.path.join(train_result_path, 'log.csv'),
                    mode='a',
                    index=False,
                    header=False,
                )
                torch.save(
                    {
                        'epoch': epoch,
                        'model': model.state_dict(),
                        'opt': optimizer.state_dict(),
                    },
                    os.path.join(models_save_path, "%d_%.4f.pt" % (epoch, test_acc)),
                )
