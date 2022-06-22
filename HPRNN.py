
import torch
import torch.nn as nn
from configs import Configs
from models.ConvLSTMCell import ConvLSTMCell
from models.SaConvlstmCell import SAConvLSTMCell
from models.ConvGRU import ConvGRUCell
from models.CNNCell import EncoderCNNCell, DecoderCNNCell
from models.IntegrationCell import IntegrationCell,GuidingCell
from models.RefinementCell import RefinementCell


class HPRNN(nn.Module):
    def __init__(self, nf):
        super(HPRNN, self).__init__()

        """ ARCHITECTURE 
        # Encoder (ConvLSTM)
        # Encoder Vector (final hidden state of encoder)
        # Decoder (ConvLSTM) - takes Encoder Vector as input
        # Decoder (3D CNN) - produces regression predictions for our model
        """
        self.encoder_cnn = EncoderCNNCell()
        self.encoder_1_convlstm = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                            #    d_attn=d_attn,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.encoder_2_convlstm = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                            #    d_attn=d_attn,
                                               kernel_size=(3, 3),
                                               bias=True)
        self.encoder_integration = IntegrationCell()

        self.encoder_long_range_rnn = ConvGRUCell(input_dim=nf*2,
                                         hidden_dim=nf*2,
                                         kernel_size=(3, 3),
                                         bias=True)

        self.decoder_long_range_rnn = ConvGRUCell(input_dim=nf*2,
                                         hidden_dim=nf*2,
                                         kernel_size=(3, 3),
                                         bias=True)

        self.decoder_Guiding = GuidingCell()
        self.decoder_1_convlstm = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                            #    d_attn=d_attn,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_2_convlstm = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                            #    d_attn=d_attn,
                                               kernel_size=(3, 3),
                                               bias=True)     
        self.decoder_refinement = RefinementCell(input_dim=nf,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)   
        self.decoder_cnn = DecoderCNNCell()

    def autoencoder(self, x, seq_len, future_step, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4, h_tl, c_tl): # m_t, m_t2, m_t3, m_t4):

        outputs = []
        slide_window = 3
        hidden_slide =[]
        integration_input = torch.Tensor([]).cuda()
        c = 1
        # encoder 
        print("In ENCODER!!")     
        for t in range(seq_len):
            print("seq_",t)         
            cnn_out = self.encoder_cnn(x[:, t, :, :, :])
            
            h_t, c_t = self.encoder_1_convlstm(input_tensor=cnn_out,
                                               cur_state=[h_t, c_t])  # we could concat to provide skip conn here
            h_t2, c_t2 = self.encoder_2_convlstm(input_tensor=h_t,
                                                 cur_state=[h_t2, c_t2])  # we could concat to provide skip conn here
            # h_t, c_t, m_t = self.encoder_1_convlstm(input_tensor=cnn_out,
            #                                    cur_state=[h_t, c_t, m_t])  # we could concat to provide skip conn here
            # h_t2, c_t2, m_t2 = self.encoder_2_convlstm(input_tensor=h_t,
            #                                      cur_state=[h_t2, c_t2, m_t])  # we could concat to provide skip conn here
            
            hidden_slide.append(h_t2.unsqueeze(1))            
            if (c == slide_window):                      
                concate_n = torch.cat(hidden_slide, 1)
                integration_input = torch.cat((integration_input, concate_n.unsqueeze(0)), 0)                              
                hidden_slide = []
                c = 0 
            c+= 1
        # print("cnn_out",cnn_out.size()) 
        # print("integration_input",integration_input.size()) 

        encoder_vector = torch.Tensor([]).cuda()
        for k in range(integration_input.size(0)):
            integration = self.encoder_integration(integration_input[k,:,:,:,:,:]).unsqueeze(0)
            encoder_vector = torch.cat([encoder_vector,integration], 0)
        
        # encoder_vector = h_t2
        # print("encoder_vector_out",encoder_vector.size())

        # encoder - convGRU
        
        h_tl_cat_encoder = torch.Tensor([]).cuda()
        print("convGRU")
        for i in range(encoder_vector.size(0)):
            encoder_input = encoder_vector[i,:,:,:,:,:] #第幾組  
            h_tl_out = h_tl   
            h_tl_encoder = torch.Tensor([]).cuda()          
            for j in range(encoder_vector.size(2)): #這組的時間步
                h_tl_out = self.encoder_long_range_rnn(input_tensor=encoder_input[:,j,:,:,:],
                                                h_cur=h_tl_out)
                h_tl_encoder = torch.cat([h_tl_encoder,h_tl_out.unsqueeze(0)], 0)
            h_tl_cat_encoder = torch.cat([h_tl_cat_encoder,h_tl_encoder.unsqueeze(0)], 0)
        # print("h_tl_cat",h_tl_cat_encoder.size())  
        
        h_t3 = h_t2
        c_t3 = c_t2
        h_encoder = h_t2
        c_encoder = c_t2
        
        # decoder
        print("In DECODER!!")
        for n in range(future_step//6):
            # print("==hour==",n)
            
            h_tl_decoder = torch.Tensor([]).cuda()        
            h_tl_out_decoder = h_tl_out
            for i in range(h_tl_cat_encoder.size(0)):
                decoder_input = h_tl_cat_encoder[i,:,:,:,:,:] #第幾組  
                for j in range(h_tl_cat_encoder.size(1)): #這組的時間步
                    h_tl_out_decoder = self.decoder_long_range_rnn(input_tensor=decoder_input[j,:,:,:,:],
                                                    h_cur=h_tl_out_decoder)
                    h_tl_decoder = torch.cat([h_tl_decoder,h_tl_out_decoder.unsqueeze(0)], 0)
            # print("h_tl_decoder_output",h_tl_decoder.size())              
            guiding_out = self.decoder_Guiding(h_tl_decoder)
            # print("guiding_out",guiding_out.size())
            

            for t in range(guiding_out.size(0)):
                h_t3, c_t3 = self.decoder_1_convlstm(input_tensor=guiding_out[t,:,:,:,:],
                                                    cur_state=[h_t3, c_t3])  # we could concat to provide skip conn here
                h_t4, c_t4 = self.decoder_2_convlstm(input_tensor=h_t3,
                                                    cur_state=[h_t4, c_t4])  # we could concat to provide skip conn here
                # h_t3, c_t3, m_t3 = self.decoder_1_convlstm(input_tensor=guiding_out[t,:,:,:,:],
                #                                     cur_state=[h_t3, c_t3, m_t2])  # we could concat to provide skip conn here
                # h_t4, c_t4, m_t4 = self.decoder_2_convlstm(input_tensor=h_t3,
                #                                     cur_state=[h_t4, c_t4, m_t3])  # we could concat to provide skip conn here
                o, h_encoder, c_encoder= self.decoder_refinement(h_t4,[h_encoder,c_encoder])
                o = self.decoder_cnn(o)          
                outputs.append(o)  # predictions
            # print("decoder_out",o.size())
        

        outputs = torch.stack(outputs, 0).permute(1, 0, 2, 3, 4).contiguous()
        print("output",outputs.size())
        return outputs

    def forward(self, x, future_seq=12, hidden_state=None):

        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor of shape (b, t, c, h, w)        #   batch, time, channel, height, width
        """

        # find size of different input dimensions
        b, seq_len, _, h, w = x.size()
        img_size = h // 4
        # initialize hidden states
        h_t, c_t = self.encoder_1_convlstm.init_hidden(batch_size=b, image_size=(img_size, img_size))
        h_t2, c_t2 = self.encoder_2_convlstm.init_hidden(batch_size=b, image_size=(img_size, img_size))
        h_t3, c_t3 = self.decoder_1_convlstm.init_hidden(batch_size=b, image_size=(img_size, img_size))
        h_t4, c_t4 = self.decoder_2_convlstm.init_hidden(batch_size=b, image_size=(img_size, img_size))
        h_tl,c_tl = self.encoder_long_range_rnn.init_hidden(batch_size=b, image_size=(img_size, img_size))
        # h_t, c_t, m_t = self.encoder_1_convlstm.init_hidden(batch_size=b, image_size=(img_size, img_size))
        # h_t2, c_t2, m_t2 = self.encoder_2_convlstm.init_hidden(batch_size=b, image_size=(img_size, img_size))
        # h_t3, c_t3, m_t3 = self.decoder_1_convlstm.init_hidden(batch_size=b, image_size=(img_size, img_size))
        # h_t4, c_t4, m_t4 = self.decoder_2_convlstm.init_hidden(batch_size=b, image_size=(img_size, img_size))
        # h_tl,c_tl = self.encoder_long_range_rnn.init_hidden(batch_size=b, image_size=(img_size, img_size))

        # autoencoder forward
        outputs = self.autoencoder(x, seq_len, future_seq, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4, h_tl, c_tl)# m_t, m_t2, m_t3, m_t4)

        return outputs