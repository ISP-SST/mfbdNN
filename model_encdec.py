import torch
import torch.nn as nn
import torch.utils.data

class ConvBlock(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size=3, stride=1, upsample=False):
        super(ConvBlock, self).__init__()

        self.upsample = upsample

        if (upsample):
            self.upsample = nn.Upsample(scale_factor=2)
            self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=1)
        else:
            self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride)

        nn.init.kaiming_normal_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0.1)

        self.reflection = nn.ReflectionPad2d(int((kernel_size-1)/2))
        self.bn = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn(x)
        out = self.relu(out)
        
        if (self.upsample):
            out = self.upsample(out)

        out = self.reflection(out)
        out = self.conv(out)
            
        return out

class deconv_block(nn.Module):
    def __init__(self, n_frames):
        super(deconv_block, self).__init__()

        nfiltro = 8
        self.A01 = ConvBlock(n_frames, nfiltro, kernel_size=3)
        
        self.C01 = ConvBlock(nfiltro, 2*nfiltro, stride=2)
        self.C02 = ConvBlock(2*nfiltro, 2*nfiltro)
        self.C03 = ConvBlock(2*nfiltro, 2*nfiltro)
        self.C04 = ConvBlock(2*nfiltro, 2*nfiltro, kernel_size=1)

        self.C11 = ConvBlock(2*nfiltro, 2*nfiltro)
        self.C12 = ConvBlock(2*nfiltro, 2*nfiltro)
        self.C13 = ConvBlock(2*nfiltro, 2*nfiltro)
        self.C14 = ConvBlock(2*nfiltro, 2*nfiltro, kernel_size=1)
        
        self.C21 = ConvBlock(2*nfiltro, 4*nfiltro, stride=2)
        self.C22 = ConvBlock(4*nfiltro, 4*nfiltro)
        self.C23 = ConvBlock(4*nfiltro, 4*nfiltro)
        self.C24 = ConvBlock(4*nfiltro, 4*nfiltro, kernel_size=1)
        
        self.C31 = ConvBlock(4*nfiltro, 8*nfiltro, stride=2)
        self.C32 = ConvBlock(8*nfiltro, 8*nfiltro)
        self.C33 = ConvBlock(8*nfiltro, 8*nfiltro)
        self.C34 = ConvBlock(8*nfiltro, 8*nfiltro, kernel_size=1)
        
        self.C41 = ConvBlock(8*nfiltro, 4*nfiltro, upsample=True)
        self.C42 = ConvBlock(4*nfiltro, 4*nfiltro)
        self.C43 = ConvBlock(4*nfiltro, 4*nfiltro)
        self.C44 = ConvBlock(4*nfiltro, 4*nfiltro)
        
        self.C51 = ConvBlock(4*nfiltro, 2*nfiltro, upsample=True)
        self.C52 = ConvBlock(2*nfiltro, 2*nfiltro)
        self.C53 = ConvBlock(2*nfiltro, 2*nfiltro)
        self.C54 = ConvBlock(2*nfiltro, 2*nfiltro)
        
        self.C61 = ConvBlock(2*nfiltro, 2*nfiltro, upsample=True)
        self.C62 = ConvBlock(2*nfiltro, 2*nfiltro)
        self.C63 = ConvBlock(2*nfiltro, int(nfiltro/2))

        self.C64 = nn.Conv2d(int(nfiltro/2), 1, kernel_size=1, stride=1)
        # self.C65 = nn.Conv2d(7, 1, kernel_size=1, stride=1)

        nn.init.kaiming_normal_(self.C64.weight)
        nn.init.constant_(self.C64.bias, 0.1)
        
        # nn.init.kaiming_normal_(self.C65.weight)
        # nn.init.constant_(self.C65.bias, 0.1)
                
        
    def forward(self, x):        
        A01 = self.A01(x)

        # N -> N/2
        C01 = self.C01(A01)
        C02 = self.C02(C01)
        C03 = self.C03(C02)
        C04 = self.C04(C03)
        C04 += C01
        
        # N/2 -> N/2
        C11 = self.C11(C04)
        C12 = self.C12(C11)
        C13 = self.C13(C12)
        C14 = self.C14(C13)
        C14 += C11
        
        # N/2 -> N/4
        C21 = self.C21(C14)
        C22 = self.C22(C21)
        C23 = self.C23(C22)
        C24 = self.C24(C23)
        C24 += C21
        
        # N/4 -> N/8
        C31 = self.C31(C24)
        C32 = self.C32(C31)
        C33 = self.C33(C32)
        C34 = self.C34(C33)
        C34 += C31
        
        C41 = self.C41(C34)
        C41 += C24
        C42 = self.C42(C41)
        C43 = self.C43(C42)
        C44 = self.C44(C43)
        C44 += C41
        
        C51 = self.C51(C44)
        C51 += C14
        C52 = self.C52(C51)
        C53 = self.C53(C52)
        C54 = self.C54(C53)
        C54 += C51
        
        C61 = self.C61(C54)        
        C62 = self.C62(C61)
        C63 = self.C63(C62)
        C64 = self.C64(C63)
        # C65 = self.C65(x)
        # out = C64 + C65
        out = C64 + x[:,0:1,:,:]
        
        return out