from configs import configs
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from HPRNN import HPRNN
from radar_echo_p20_muti_sample_drop_08241800_load_512x512 import load_data
from visualize.Verification import Verification 
from visualize.visualized_pred import visualized_area_with_map

def test(model, model_name, save_path, itr, device, test_loader, data_name):
    model_path = os.path.join(save_path, model_name)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_loss = 0
    save_test_path = save_path + 'test_itr_{}_csi_{}/'.format(itr,data_name)
    if not os.path.isdir(save_test_path):
        os.makedirs(save_test_path) 
    with torch.no_grad():
        num_of_batch_size = test_loader.step_per_epoch
        for index in range(num_of_batch_size):
            place = "Sun_Moon_Lake"
            data, target = test_loader.generator_getClassifiedItems_3(index, place)
            data = torch.FloatTensor(data).to(device) 
            target = torch.FloatTensor(target).to(device)
            output = model(data)
            # test_loss += F.mse_loss(output, target,reduction='mean').item() 
            output = output.reshape(-1,configs.output_length,512,512,1)
            print("output",output.shape)
            target = target.reshape(-1,configs.output_length,512,512,1)
            print("target",target.shape)
            sum_mse = 0
            new_gt = []
            new_output = []
            img_size = 128
            for i in range(configs.output_length):
                    vis_gx = np.array(torch.squeeze(output[:, i, :, :, :]).cpu())
                    vis_x = np.array(torch.squeeze(target[:, i, :, :, :]).cpu())

                    #重新以中心點取往外的大小                    
                    a=int(512/2-img_size/2)
                    b=int(512/2+img_size/2)
                    vis_gx=vis_gx[a:b,a:b]
                    vis_x=vis_x[a:b,a:b]                       
                    vis_gx[vis_gx <= 1] = 0    # clear dbz < 1  
                    new_gt.append(vis_x)
                    new_output.append(vis_gx)
                    visualized_area_with_map(vis_gx, 'Sun_Moon_Lake', shape_size=[128,128], title='vis_pred_{}'.format(i), savepath=save_test_path)
                    visualized_area_with_map(vis_x, 'Sun_Moon_Lake', shape_size=[128,128], title='vis_gt_{}'.format(i), savepath=save_test_path)   
                    # fn = save_test_path+'_sqe{}.txt'.format(i)
                    # np.savetxt(save_test_path+'output_{}.csv'.format(i), vis_gx, delimiter = ',')
                    mse = np.square(vis_x - vis_gx).sum()
                    mse = mse/(64*64)
                    sum_mse += mse
                    # with open(fn,'a') as file_obj:
                    #     file_obj.write('output[{}]_mse={}' .format(i,str(mse))+'\n')
                    #     file_obj.write('output[{}]_rmse={}' .format(i,str(np.sqrt(mse)))+'\n')
                        
    new_gt = np.array(new_gt).reshape(-1,configs.output_length,128,128,1)  
    print(new_gt.shape)              
    new_output = np.array(new_output).reshape(-1,configs.output_length,128,128,1)    
    print(new_output.shape)            
    test_loss = sum_mse / configs.output_length
    print('\nTest set: Average loss: {:.4f}'.format(test_loss))
    
    #局部找過山個案
    rmse = np.sqrt(((new_gt-new_output)**2).mean())
    rmse_1th = np.sqrt(((new_gt[:, :6, :, :, :]-new_output[:, :6, :, :, :])**2).mean())
    rmse_2th = np.sqrt(((new_gt[:, 6:, :, :, :]-new_output[:, 6:, :, :, :])**2).mean())
    fn = save_test_path + '{}_rmse.txt'.format(data_name)
    with open(fn,'a') as file_obj:
        file_obj.write('rmse=' + str(rmse)+'\n')
        file_obj.write('rmse1th=' + str(rmse_1th)+'\n')
        file_obj.write('rmse2th=' + str(rmse_2th)+'\n')
    #原本
    # test_loss /= num_of_batch_size
    # print('\nTest set: Average loss: {:.4f}'.format(test_loss))
    # output = np.array(output.cpu())
    # target = np.array(target.cpu())
    # rmse=np.sqrt(((output - target) ** 2).mean())
    # rmse_1th=np.sqrt(((output[:, :6, :, :, :] - target[:, :6, :, :, :]) ** 2).mean())
    # rmse_2th=np.sqrt(((output[:, 6:, :, :, :] - target[:, 6:, :, :, :]) ** 2).mean())
    # print(output[:, :6, :, :, :].shape)
    # print(output[:, 6:, :, :, :].shape)
    # fn = save_test_path + '{}_rmse.txt'.format(data_name)
    # with open(fn,'a') as file_obj:
    #     file_obj.write('rmse=' + str(rmse)+'\n')
    #     file_obj.write('rmse1th=' + str(rmse_1th)+'\n')
    #     file_obj.write('rmse2th=' + str(rmse_2th)+'\n')
    
    csi_picture(img_out = new_output,test_ims= new_gt,save_path = save_test_path+'csi_{}/'.format(data_name),data_name=data_name)
    
    #直接計算CSI
    # csi = np.genfromtxt('D:/radar/HPRNN/save_model/bmse/test_itr_160_csi_202005270000to12/csi_202005270000to12/202005270000_07to12.csv',delimiter=',')
    # data_name = "20200527_07to12"
    # draw_CSI(csi, np.array(output.cpu()),np.array(target.cpu()),data_name,save_test_path+'csi_202005270000to12/')
    
def draw_CSI(csi, data_name, save_path):
        ## Draw thesholds CSI
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
        ax.set_facecolor((229.0/255.0, 229.0/255.0, 229.0/255.0))
        plt.xlim(0, 60)
        plt.ylim(-0.05, 1.0)
        plt.xlabel('Threshold')
        plt.ylabel('CSI')
        plt.title('{}\nThresholds CSI'.format(data_name))
        plt.grid(True)
        for period in range(configs.input_length):
            plt.plot(np.arange(csi.shape[1]), [np.nan] + csi[period, 1:].tolist(), linewidth=2.0, label='{} min'.format((period+1)*10))

        plt.legend(loc='upper right')
        fig.savefig(fname=save_path+'Thresholds_CSI_ALL.png', format='png')
        plt.clf()

        ## Draw thesholds AVG CSI
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
        ax.set_facecolor((229.0/255.0, 229.0/255.0, 229.0/255.0))
        plt.xlim(0, 60)
        plt.ylim(-0.05, 1.0)
        plt.xlabel('Threshold')
        plt.ylabel('CSI')
        plt.title('{}\nThresholds CSI'.format(data_name))
        plt.grid(True)
        plt.plot(np.arange(csi.shape[1]), [np.nan] + np.mean(csi[:, 1:], 0).tolist(), linewidth=2.0, label='AVG CSI')   
        plt.legend(loc='upper right')
        fig.savefig(fname=save_path+'Thresholds_AVG_CSI.png', format='png')
        plt.clf()

def csi_picture(img_out, test_ims, save_path,data_name='csi'):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)       
        ## CSI comput
        csi = []
        for period in range(configs.output_length):
            print("period=",period)
            csi_eva = Verification(pred=img_out[:, period].reshape(-1, 1), target=test_ims[:, period].reshape(-1, 1), threshold=60, datetime='')
            print("csi_eva.csi.shape=",np.array(csi_eva.csi).shape)# (60, 99225)
            csi.append(np.nanmean(csi_eva.csi, axis=1))
            # print("csi",csi)
            print("mean",np.mean(csi_eva.csi[0,np.isfinite(csi_eva.csi[0,:])]))
            print("np.array(csi).shape=",np.array(csi).shape)#(1, 60)
            # sys.exit()
        
        csi = np.array(csi)
        np.savetxt(save_path+'{}.csv'.format(data_name), csi, delimiter = ',')
        np.savetxt(save_path+'{}_01to06.csv'.format(data_name), csi[:6,], delimiter = ',')
        np.savetxt(save_path+'{}_07to12.csv'.format(data_name), csi[6:,], delimiter = ',')
        # np.savetxt(save_path+'T202005270000csi.csv', csi.reshape(6,60), delimiter = ' ')
        draw_CSI(csi, data_name, save_path)

def main():
    torch.manual_seed(configs.seed)
    device = configs.device
    save_path = configs.save_path
    load_radar_echo_df_path = None #"D:/radar/HPRNN/202108061300to12(Sun_Moon_Lake)_128_128x128.pkl"
    radar_echo_storage_path = "I:/radar/20210606and0806data/"
    data_name = "202005270500to12(Sun_Moon_Lake)"
    test_date=[['2020-05-27 05:00', '2020-05-27 05:01']]
    date_date=[['2020-05-27 05:00', '2020-05-27 05:01']]
    # data_name = "202008252000to12"
    # test_date=[['2020-08-25 20:10', '2020-08-25 20:11']]
    # date_date=[['2020-08-25 20:10', '2020-08-25 20:11']]
    # data_name = '202106051000to12'
    # date_date=[['2021-06-05 10:10','2021-06-05 10:11']]
    # test_date=[['2021-06-05 10:10','2021-06-05 10:11']]
    # data_name = '202106060800to12(Sun_Moon_Lake)'
    # date_date=[['2021-06-06 08:10','2021-06-06 08:11']]
    # test_date=[['2021-06-06 08:10','2021-06-06 08:11']]
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
        date_range= date_date,
        test_date= test_date,
        save_np_radar=save_path
    )
    if not load_radar_echo_df_path:
        radar.exportRadarEchoFileList()
        radar.saveRadarEchoDataFrame(path=configs.save_path ,load_name_pkl='{}_128x128'.format(data_name))
    test_loader = radar.generator('test', batch_size=1, save_path=save_path)
    model = HPRNN(64).to(device)
    print(model)
    model_name = "model40_loss27.37463791514703.pkl"
    save_path = save_path+"/save_model/weight1.2.5/"
    itr = 40
    test(model, model_name, save_path, itr, device, test_loader, data_name)


if __name__ == '__main__':
    main()