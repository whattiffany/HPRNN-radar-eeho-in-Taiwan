from configs import configs
import datetime
from time import time
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from HPRNN import HPRNN
# from HPRNN_attn import HPRNN
from loss import GAILossMD
from radar_echo_p20_muti_sample_drop_08241800_load_512x512 import load_data
NOISE_SIGMA = 1.0
class MyMSELoss(torch.nn.Module):
  def __init__(self, weight):
    super(MyMSELoss, self).__init__()
    self.weight =  weight
   
  def forward(self, output, label):
    # print("==================")
    # error = output - label#!
    label = label.float() 
    error = (label - output)**2#!
    # print(output - label)
    '''
    依照label的dBz區間 給予誤差不同權重
    '''
    print("weight",self.weight)
    error_weight = torch.where((label < 40), error*self.weight[0], error)

    # error_weight = torch.where((label >= 22) & (label < 28), torch.pow(error,2)*self.weight[1], error_weight)

    # error_weight = torch.where((label >= 28) & (label < 33), torch.pow(error,2)*self.weight[2], error_weight)
    # error_weight = torch.where((label >= 33) & (label < 40), torch.pow(error,2)*self.weight[1], error_weight)   
    # error_weight = torch.where((label > 28) &(label <= 40), error_weight*self.weight[1], error_weight)
    error_weight = torch.where((label >= 40) & (label < 45), error*self.weight[1], error_weight) 
    error_weight = torch.where((label >= 45), error*self.weight[2], error_weight) 
    # error_weight = torch.where((label >= 45) & (label < 50), torch.pow(error,2)*self.weight[5], error_weight) 
    # error_weight = torch.where((label >= 50), torch.pow(error,2)*self.weight[6], error_weight) 
    error_weight_mean = torch.mean(error_weight)

    return error_weight_mean

def train(configs, model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss_sum = 0
    num_of_batch_size = train_loader.step_per_epoch-1
    start = time()
    weight = [1,2,5]#!
    for index in range(num_of_batch_size):
        data, target = train_loader.generator_getClassifiedItems_3(index, "Sun_Moon_Lake")
        data = torch.FloatTensor(data).to(device) 
        target = torch.FloatTensor(target).to(device)
        optimizer.zero_grad()
        print("data",data.size()) 
        print("target",target.size())
        output = model(data)
        # loss = F.l1_loss(output, target)+F.mse_loss(output, target)
        # loss = F.mse_loss(output, target)
        loss = F.l1_loss(output, target)+MyMSELoss(weight)(output, target)
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
    weight = [1,2,5]#!
    with torch.no_grad():
        num_of_batch_size = test_loader.step_per_epoch-1
        for index in range(num_of_batch_size):
            data, target = test_loader.generator_getClassifiedItems_3(index, "Sun_Moon_Lake")
            data = torch.FloatTensor(data).to(device) 
            target = torch.FloatTensor(target).to(device)
            output = model(data)
            # test_loss += F.mse_loss(output, target).item() 
            test_loss += (F.l1_loss(output, target)+MyMSELoss(weight)(output, target)).item() 
            # test_loss += (F.l1_loss(output, target)+F.mse_loss(output, target)).item() 
    test_loss /= num_of_batch_size

    print('\nTest set: Average loss: {:.4f}'.format(test_loss))
    return test_loss


def main(transfer=False):
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(configs.seed)
    device = configs.device

    train_kwargs = {'batch_size': configs.batch_size}
    test_kwargs = {'batch_size': configs.batch_size_test}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # load_radar_echo_df_path ="D:/radar/2014to2020_512_2hr.pkl"
    load_radar_echo_df_path ="E:/radar/up15dBZ_2020_2021.pkl"
    save_path = configs.save_path
    radar_echo_storage_path = "D:/radar/NWP/2022_03_20_27_OBS/2022_03_20_27_OBS/"
    radar = load_data(
        radar_echo_storage_path=radar_echo_storage_path,
        # 'data/RadarEcho_Bao_Zhong_2018_08240010_T6toT6_inoutputshape64_random.pkl',#None,#load_radar_echo_df_path,
        load_radar_echo_df_path=load_radar_echo_df_path,
        input_shape=[512, 512],
        output_shape=[512, 512],
        period=configs.input_length,
        predict_period=configs.output_length,
        places=["Sun_Moon_Lake"],
        random=False,
        date_range= configs.new_train_date_2,
        test_date= configs.test_date,
        save_np_radar=save_path
    )
    if not load_radar_echo_df_path:
        radar.exportRadarEchoFileList()
        # radar.saveRadarEchoDataFrame(path=configs.save_path ,load_name_pkl='{}_512x512'.format("20220320to0327"))
    train_loader = radar.generator(
        'train', batch_size=configs.batch_size, save_path=save_path)
    test_loader = radar.generator(
        'val', batch_size=configs.batch_size, save_path=save_path)

    model = HPRNN(64).to(device)
    # model = HPRNN(64,configs.attn).to(device)
    if transfer == True:
        model.load_state_dict(torch.load("D:/radar/HPRNN/save_model/bmse+CBAM+weight(1.2.5)/model104_loss32.70523884324881.pkl"))
        # frozen = [
            #     # "decoder_1_convlstm.conv.weight",
            #     # "decoder_1_convlstm.conv.bias",
            #     # "decoder_1_convlstm.conv_LN.weight",
            #     # "decoder_1_convlstm.conv_LN.bias",
            #     # "decoder_2_convlstm.conv.weight",
            #     # "decoder_2_convlstm.conv.bias",
            #     # "decoder_2_convlstm.conv_LN.weight",
            #     # "decoder_2_convlstm.conv_LN.bias",
            #     "decoder_refinement.conv.weight",
            #     "decoder_refinement.conv.bias",
            #     "decoder_refinement.residual_block.conv.weight",
            #     "decoder_refinement.residual_block.conv.bias",
            #     "decoder_refinement.residual_block.ca.fc.0.weight",
            #     "decoder_refinement.residual_block.ca.fc.2.weight",
            #     "decoder_refinement.residual_block.sa.conv1.weight",
            #     "decoder_refinement.convlstm.conv.weight",
            #     "decoder_refinement.convlstm.conv.bias",
            #     "decoder_refinement.convlstm.conv_LN.weight",
            #     "decoder_refinement.convlstm.conv_LN.bias",
            #     "decoder_cnn.deconv.0.weight",
            #     "decoder_cnn.deconv.0.bias",
            #     "decoder_cnn.deconv.2.weight",
            #     "decoder_cnn.deconv.2.bias",
            #     "decoder_cnn.deconv.4.weight",
            #     "decoder_cnn.deconv.4.bias"]
        # for name,param in model.named_parameters():
        #     # print(name)
        #     if name not in frozen:
        #         # print(name)
        #         param.requires_grad = False
        #     else:
        #         print("not frozen",name)
        # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=configs.lr)
        optimizer = optim.Adam(model.parameters(), lr=configs.lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=configs.lr)
    # scheduler = StepLR(optimizer, step_size=20, gamma=configs.gamma)

    time = datetime.datetime.now()
    loss_file = save_path + 'loss_newtrans_(1_2_5).txt'
    with open(loss_file,'a') as file_obj:
        file_obj.write(str(time)+"\n")
    
    min_loss = 200  
    trigger_times = 0
    patience = 10

    for epoch in range(1, configs.epochs + 1):
        print("Epoch now: ",epoch)
        
        train_loss = train(configs, model, device, train_loader, optimizer, epoch)
        test_loss = test(model, device, test_loader)
        # scheduler.step()
        loss_file = save_path + 'loss_newtrans_(1_2_5).txt'
        with open(loss_file,'a') as file_obj:
            file_obj.write("-----itr ="+str(epoch)+"----- \n")
            # file_obj.write('args.total_length - args.input_length =' + str(12 - args.input_length)+'\n')
            file_obj.write("model train loss " + str(train_loss)  + '\n' )
            file_obj.write("model test loss " + str(test_loss)  + '\n')

        if (epoch >=10):
            print("saving model...")
            path = os.path.join(save_path+'save_model/', 'model{}_loss{}.pkl'.format(str(epoch), str(train_loss)))
            torch.save(model.state_dict(),path)

        if test_loss > min_loss:
            trigger_times += 1
            print('trigger times:', trigger_times)
            
            if trigger_times >= patience:
                print('Early stopping!\nStart to test process.')
                # model_pkl ='_itr{}_earlystopping{}_test_cost{}_min_loss{}.pkl'.format(str(epoch), patience, 
                # test_loss, min_loss)
                # torch.save(model_pkl,save_path)
                return 0
        else:
            trigger_times = 0
            print('trigger times: ',trigger_times)

            min_loss = test_loss

if __name__ == '__main__':
    main(transfer=True)