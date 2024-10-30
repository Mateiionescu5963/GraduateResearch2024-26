import torch
from torch.optim import Adam, lr_scheduler
from torch.autograd import Variable
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from utils import save_model, calculate_result

def eval_model(model, dataloader, device):
    print('Start evaluate...')
    model.eval()
    with torch.no_grad():
        f1_score_total = 0.0
        precision_total = 0.0
        recall_total = 0.0
        accuracy_total = 0.0
        
        for step, data in enumerate(dataloader):
            labels = data[1].to(device)
            inputs = data[0].to(device)
            outputs = model(inputs)
            acc, pre, rec, f1 = calculate_result(outputs, labels)
            f1_score_total += f1
            precision_total += pre
            recall_total += rec
            accuracy_total += acc
        
        f1_score_total /= len(dataloader)
        precision_total /= len(dataloader)
        recall_total /= len(dataloader)
        accuracy_total /= len(dataloader)
        
    return accuracy_total, precision_total, recall_total, f1_score_total

def train_model(model, criterion, optimizer, device, epochs, train_loader, valid_loader):
    print('Start training...')

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)
    dataset_size = len(train_loader)

    best_model = model
    best_f1 = 0.0

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))

        epoch_loss = 0.0

        for step, batch_data in enumerate(train_loader):
            print("-----------------")
            print("Step is : ", step)

            model.train()
            optimizer.zero_grad()

            exe_input = batch_data[0].to(device).long()
            label = batch_data[1].to(device).float()

            outputs = model(exe_input)

            loss = criterion(outputs, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            acc, pre, rec, f1 = calculate_result(outputs, label)
            print("accuracy is {:.3f}, precision is {:.3f}, recall is {:.3f}, f1 is {:.3f}, loss is {:.3f}".format(acc, pre, rec, f1, loss))

        epoch_loss /= dataset_size
        print('Epoch {:<4} Loss: {:.4f}'.format(epoch, epoch_loss))

        scheduler.step(epoch_loss)

    acc, pre, rec, f1 = eval_model(model, valid_loader, device)
    print('Epoch accuracy is {:.3f}, precision is {:.3f}, Recall is {:.3f}, F1 is {:.3f}.'.format(acc, pre, rec, f1))
    
    if f1 >= best_f1:
        best_f1 = f1
        best_model = model
        save_model(model, 'best_model_final.pt')

    return best_model
