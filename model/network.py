import torch
import torch.nn as nn
import torch.nn.functional as F

from model.M_Net import M_net
from model.T_Net_psp import PSPNet

class net_T(nn.Module):
    # Train T_net
    def __init__(self):

        super(net_T, self).__init__()

        self.t_net = PSPNet()

    def forward(self, input):

    	# trimap
        """
        这里trimap是没有加softmax的，没问题
        """
        trimap= self.t_net(input)
        return trimap

class net_M(nn.Module):
    '''
		train M_net
    '''

    def __init__(self):

        super(net_M, self).__init__()
        self.m_net = M_net()

    def forward(self, input, trimap):

        # paper: bs, fs, us
        bg, fg, unsure = torch.split(trimap, 1, dim=1)
        
        # concat input and trimap
        m_net_input = torch.cat((input, trimap), 1)

        # matting
        alpha_r = self.m_net(m_net_input)
        # fusion module
        # paper : alpha_p = fs + us * alpha_r
        alpha_p = fg + unsure * alpha_r

        return alpha_p

class net_F(nn.Module):
    '''
		end to end net 
    '''

    def __init__(self):

        super(net_F, self).__init__()

        self.t_net = PSPNet()
        self.m_net = M_net()

    def forward(self, input):

    	# trimap
        trimap = self.t_net(input)
        """
            F.softmax：对n维输入张量运用Softmax函数，将张量的每个元素缩放到（0,1）区间且和为1
            dim=1 ：在通道(c)方向进行softmax
        """
        trimap_softmax = F.softmax(trimap, dim=1)

        # paper: bs, fs, us
        """
            torch.split(tensor=trimap_softmax , split_size=1, dim=1)
            其中：dim=1:在通道(c)方向进行
                 split_size=1：通道数为3，每个块的大小为1
            对trimap_softmax，按通道方向进行分离，得到的第一个张特征图代表bg的概率图，第二张特征图为代表unsure的概率图
            第三张特征图为代表fg的概率图
        """
        # 疑问：bg, unsure, fg都只是概率图，不需要进行argmax，使其均0/1化么？
        bg, unsure, fg = torch.split(trimap_softmax, 1, dim=1)

        # concat input and trimap
        # 疑问：trimap_softmax，没有进行argmax呀？
        m_net_input = torch.cat((input, trimap_softmax), 1)

        # matting

        alpha_r = self.m_net(m_net_input)
        # fusion module
        # paper : alpha_p = fs + us * alpha_r
        # 疑问：fg只是概率图，并不是0/1 ； unsure只是概率图，并不是0/1
        alpha_p = fg + unsure * alpha_r

        return trimap, alpha_p
