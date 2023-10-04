import csv
import torch
from pytorchtools import EarlyStopping
from metrics import WeightMSE, RMSE, SSIM
import numpy as np
from model import CNN_Attention

def save_file(fn, data):
    with open(fn, 'w', newline='') as f:
        csv_write = csv.writer(f)
        csv_write.writerows(data)
    f.close()


def train_model(model, epochs, optimizer, lr_scheduler, train_loader, val_loader, save_model, save_path, device='cuda'):
    avg_train_losses, avg_valid_losses = [], []
    img_size = 312
    weight = [1, 5, 10, 15, 20]
    early_stopping = EarlyStopping(patience=10, verbose=True, path=save_model + 'checkpoint.pt')
    custom_criterion = WeightMSE(weight)

    for epoch in range(epochs):
        train_losses, valid_losses = [], []
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.type(torch.FloatTensor).to(device)
            labels = labels.type(torch.FloatTensor).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            outputs = torch.reshape(outputs, (outputs.size(0), img_size, img_size))
            labels = torch.reshape(labels, (labels.size(0), img_size, img_size))

            loss = custom_criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        #
        model.eval()
        for i, (images, labels) in enumerate(val_loader):
            images = images.type(torch.FloatTensor).to(device)
            labels = labels.type(torch.FloatTensor).to(device)

            outputs = model(images)
            outputs = torch.reshape(outputs, (outputs.size(0), img_size, img_size))
            labels = torch.reshape(labels, (labels.size(0), img_size, img_size))

            loss = custom_criterion(outputs, labels)
            valid_losses.append(loss.item())

        avg_train_losses.append(np.average(train_losses))
        avg_valid_losses.append(np.average(valid_losses))

        if (epoch + 1) % 100 == 0:
            print("Epoch:%d, Batch:%d, Loss:%.4f" % (epoch + 1, i + 1, loss.data))

        early_stopping(np.average(valid_losses), model)
        if early_stopping.early_stop:
            print("Early Stopping")
            break

        lr_scheduler.step(np.average(train_losses))
    model.load_state_dict(torch.load(save_model + 'checkpoint.pt'))
    return model, avg_train_losses, avg_valid_losses


def test_model(model, test_loader, label_test, batch_size, save_path, device='cuda'):
    num = 0
    loss_list = []
    RMSE_loss = RMSE()
    SSIM_loss = SSIM()

    for images, labels in test_loader:
        images = images.type(torch.FloatTensor).to(device)

        model = model.eval()
        with torch.no_grad():
            outputs = model(images)

        pr = outputs.cpu().detach().numpy()
        pr = pr.reshape(pr.shape[0], 312, 312)

        labels = labels.cpu().detach().numpy()
        labels = labels.reshape(labels.shape[0], 312, 312)

        for mse_num in range(0, pr.shape[0]):
            file_name = label_test[mse_num + num * batch_size]

            pr_re = pr[mse_num, :, :]
            labels_re = labels[mse_num, :, :]

            rloss = RMSE_loss(pr_re, labels_re)
            sloss = SSIM_loss(pr_re, labels_re)

            pr_re = np.array(pr_re)
            pr_re = np.where(pr_re > 1, pr_re * 1.1, 0)
            save_loss = [file_name, str(rloss), str(sloss)]
            loss_list.append(save_loss)
            print(file_name, '%.4f' % rloss, '%.4f' % sloss)

            save_file(save_path + 'Pred_test_' + file_name + '.csv', pr_re)
            save_file(save_path + 'OBS_test_' + file_name + '.csv', labels_re)

        num = num + 1

    save_file(save_path + '_rmse.csv', loss_list)


if __name__ == '__main__':
    
    # This is just a template, please modify it according to your needs.
    train_loader = ...
    valid_loader = ...
    test_loader = ...
    label_test = ...
    
    optimizer = ...
    lr_scheduler = ...

    batch_size = 100
    epochs = 1000
    save_path = ...
    save_model = ...


    model = CNN_Attention(in_channels=20, out_channels=1, kernel_size=3, device='cuda')

    model, train_loss, valid_loss = train_model(model, epochs, optimizer, lr_scheduler, train_loader, valid_loader, save_model, save_path, device='cuda')
    test_model(model, test_loader, label_test, batch_size, save_path, device='cuda')