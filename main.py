from configs import configs
import datetime
from time import time
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from HPRNN import HPRNN
from radar_echo_p20_muti_sample_drop_08241800_load_512x512 import load_data

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss_sum = 0
    num_of_batch_size = train_loader.step_per_epoch-1
    start = time()
    for index in range(num_of_batch_size):
        data, target = train_loader.generator_getClassifiedItems_3(index, "Sun_Moon_Lake")
        data = torch.FloatTensor(data).to(device) 
        target = torch.FloatTensor(target).to(device)
        optimizer.zero_grad()
        # print("data",data.size()) 
        # print("target",target.size())
        output = model(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        train_loss_sum += loss.item()
    # if index % configs.log_interval == 0:    
        print('=== Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(
                epoch, index * len(data), num_of_batch_size,
                100. * index / num_of_batch_size, loss.item()))
    end = time()
    print("Epoch Time Cost: ",end-start)
    train_loss = train_loss_sum / num_of_batch_size
    return train_loss

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        num_of_batch_size = test_loader.step_per_epoch-1
        for index in range(num_of_batch_size):
            data, target = test_loader.generator_getClassifiedItems_3(index, "Sun_Moon_Lake")
            data = torch.FloatTensor(data).to(device) 
            target = torch.FloatTensor(target).to(device)
            output = model(data)
            test_loss += (F.mse_loss(output, target)).item()
    test_loss /= num_of_batch_size

    print('\nTest set: Average loss: {:.4f}'.format(test_loss))
    return test_loss


def main():

    torch.manual_seed(configs.seed)
    device = configs.device

    load_radar_echo_df_path = "E:/radar/2014to2020_512_2hr.pkl"
    save_path = configs.save_path
    radar_echo_storage_path = save_path
    radar = load_data(
        radar_echo_storage_path=radar_echo_storage_path,
        load_radar_echo_df_path=load_radar_echo_df_path,
        input_shape=[512, 512],
        output_shape=[512, 512],
        period=configs.input_length,
        predict_period=configs.output_length,
        places=["Sun_Moon_Lake"],
        random=False,
        date_range= configs.train_date,
        test_date= configs.test_date,
        save_np_radar=save_path
    )
    if not load_radar_echo_df_path:
        radar.exportRadarEchoFileList()
        # radar.saveRadarEchoDataFrame(path=configs.save_path ,load_ndame_pkl='{}_512x512'.format(data_name))
    train_loader = radar.generator(
        'train', batch_size=configs.batch_size, save_path=save_path)
    test_loader = radar.generator(
        'val', batch_size=configs.batch_size, save_path=save_path)

    model = HPRNN(64).to(device)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=configs.lr)
    # scheduler = StepLR(optimizer, step_size=1, gamma=configs.gamma)

    time = datetime.datetime.now()
    loss_file = save_path + 'loss.txt'
    with open(loss_file,'a') as file_obj:
        file_obj.write(str(time)+"\n")
    
    min_loss = 200  
    trigger_times = 0
    patience = 10

    for epoch in range(1, configs.epochs + 1):
        print("Epoch now: ",epoch)
        
        train_loss = train(model, device, train_loader, optimizer, epoch)
        test_loss = test(model, device, test_loader)
        # scheduler.step()

        with open(loss_file,'a') as file_obj:
            file_obj.write("-----itr ="+str(epoch)+"----- \n")
            # file_obj.write('args.total_length - args.input_length =' + str(12 - args.input_length)+'\n')
            file_obj.write("model train loss " + str(train_loss)  + '\n' )
            file_obj.write("model test loss " + str(test_loss)  + '\n')

        if (epoch >= 10):
            print("saving model...")
            path = os.path.join(save_path+'save_model', 'model{}_loss{}.pkl'.format(str(epoch), str(train_loss)))
            torch.save(model.state_dict(),path)

        if test_loss > min_loss:
            trigger_times += 1
            print('trigger times:', trigger_times)
            
            if trigger_times >= patience:
                print('Early stopping!\nStart to test process.')
                model_pkl ='_itr{}_earlystopping{}_test_cost{}_min_loss{}.pkl'.format(str(epoch), patience, 
                test_loss, min_loss)
                model.save(model_pkl,save_path)
                # return model
        else:
            trigger_times = 0
            print('trigger times: ',trigger_times)

            min_loss = test_loss


if __name__ == '__main__':
    main()