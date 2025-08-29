import numpy as np
import pandas as pd
import math
df = pd.read_csv('****.csv')
df.head(5)
# 定义区域的左下角和右上角的经纬度
left_bottom = [df['Lon_d'].min(), df['Lat_d'].min()]
right_top = [df['Lon_d'].max(), df['Lat_d'].max()]

# 定义粒度
granularity = 0.005

# 计算横向（经度）和纵向（纬度）上的网格数量
lon_grid_count = math.ceil((right_top[0] - left_bottom[0]) / granularity)
lat_grid_count = math.ceil((right_top[1] - left_bottom[1]) / granularity)

from collections import defaultdict
grid_counter = defaultdict(int)
grad_mark = {}
# 遍历轨迹数据
pos = []
sog = []
cog = []
for i in np.unique(df['mark_2']):
    data = df.loc[df['mark_2']==i][['Lon_d','Lat_d','Speed','Course']].values
    grid_representation_pos = np.zeros((lat_grid_count,lon_grid_count))
    grid_representation_sog_z = np.zeros((lat_grid_count,lon_grid_count))
    grid_representation_cog_z = np.zeros((lat_grid_count,lon_grid_count))
    grid_representation_sog = np.zeros((lat_grid_count,lon_grid_count))
    grid_representation_cog = np.zeros((lat_grid_count,lon_grid_count))
    
    grid_counter_pos = defaultdict(int)
    for point in data:
        # 确定每个点在哪个网格中
        lon_idx = int((point[0] - left_bottom[0]) / granularity)
        lat_idx = int((point[1] - left_bottom[1]) / granularity)
        
        # 增加该网格的轨迹点计数
        grid_counter_pos[(lat_idx, lon_idx)] += 1
        grid_representation_sog_z[lat_idx][lon_idx] += point[2]
        grid_representation_cog_z[lat_idx][lon_idx] += point[3]
    # 更新网格表示  
    for (lat_idx, lon_idx), count in grid_counter_pos.items():
        if count > 0:
            grid_representation_pos[lat_idx][lon_idx] = 1
            grid_representation_sog[lat_idx][lon_idx] = grid_representation_sog_z[lat_idx][lon_idx]/count
            grid_representation_cog[lat_idx][lon_idx] = grid_representation_cog_z[lat_idx][lon_idx]/count
    pos.append(grid_representation_pos)
    sog.append(grid_representation_sog/grid_representation_sog.max())
    cog.append(grid_representation_cog/grid_representation_cog.max())
pos = np.array(pos)
sog = np.array(sog)
cog = np.array(cog)
data = np.stack((pos,sog,cog), axis=1)
x_train = data
x_val = data[int(len(data)*0.8):,::]
print("x_train",x_train.shape)
print("x_val",x_val.shape)

import torch
from torch import nn
import torch.nn.functional as F
class MSAM(nn.Module):
    def __init__(self, in_channels, num_reduced_channels):
        super().__init__()
        
        self.conv1x1_1 = nn.Conv2d(in_channels, num_reduced_channels, 1, 1)
        self.conv1x1_2 = nn.Conv2d(int(num_reduced_channels*4), 1, 1, 1)
        
        self.dilated_conv3x3 = nn.Conv2d(num_reduced_channels, num_reduced_channels, 3, 1, padding=1)
        self.dilated_conv5x5 = nn.Conv2d(num_reduced_channels, num_reduced_channels, 5, 1, padding=2)
        self.dilated_conv7x7 = nn.Conv2d(num_reduced_channels, num_reduced_channels, 7, 1, padding=3)
        
    def forward(self, feature_maps):
        att = self.conv1x1_1(feature_maps)
        d1 = self.dilated_conv3x3(att)
        d2 = self.dilated_conv5x5(att)
        d3 = self.dilated_conv7x7(att)
        #print(d1.shape,d2.shape,d3.shape)
        att = torch.cat((att, d1, d2, d3), dim=1)
        att = self.conv1x1_2(att)
        return (feature_maps * att) + feature_maps

class GCAM(nn.Module):
    def __init__(self, feature_map_size, kernel_size):
        super().__init__()
        assert (kernel_size%2 == 1), "Kernel size must be odd"
        
        self.conv_q = nn.Conv1d(1, 1, kernel_size, 1, padding=(kernel_size-1)//2)
        self.conv_k = nn.Conv1d(1, 1, kernel_size, 1, padding=(kernel_size-1)//2)
        self.GAP = nn.AvgPool2d(feature_map_size)
        
    def forward(self, x):
        N, C, H, W = x.shape
        
        query = key = self.GAP(x).reshape(N, 1, C)
        query = self.conv_q(query).sigmoid()
        key = self.conv_q(key).sigmoid().permute(0, 2, 1)
        query_key = torch.bmm(key, query).reshape(N, -1)
        query_key = query_key.softmax(-1).reshape(N, C, C)
        
        value = x.permute(0, 2, 3, 1).reshape(N, -1, C)
        att = torch.bmm(value, query_key).permute(0, 2, 1)
        att = att.reshape(N, C, H, W)
        return x * att

class Fusion(nn.Module):
    def __init__(self, ch_in=9):
        super(Fusion, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        #print(x.size())
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        #beta = torch.softmax(y, dim=0)
        beta = torch.softmax(y, dim=1)
        w1 = torch.sum(beta[:,0:3], dim=1).reshape(-1, 1)
        w2 = torch.sum(beta[:,3:6], dim=1).reshape(-1, 1)
        w3 = torch.sum(beta[:,6:9], dim=1).reshape(-1, 1)
        weight = torch.cat((w1, w2, w3), dim=1).softmax(-1).reshape(-1,3,1,1,1)
        return weight

class FEM(nn.Module):
    
    def __init__(self, in_channels, num_reduced_channels, feature_map_size, kernel_size):        
        super().__init__()
        
        self.spatial_att = MSAM(in_channels, num_reduced_channels)
        self.channel_att = GCAM(feature_map_size, kernel_size)
        
        self.fusion_weights = Fusion(ch_in=9) # equal intial weights
        
    def forward(self, x):

        local_att = self.spatial_att(x)
        global_channel_att = self.channel_att(x)
        
        all_feature_maps = torch.cat((local_att, x, global_channel_att), dim=1)#[-1,9,60,100]
        local_att = local_att.unsqueeze(1)
        global_att = global_channel_att.unsqueeze(1)
        x = x.unsqueeze(1)
        feature_maps = torch.cat((local_att, x, global_att), dim=1)
        weights = self.fusion_weights(all_feature_maps)
        fused_feature_maps = (feature_maps * weights).sum(1)
        
        return fused_feature_maps

class Encoder(nn.Module):
    def __init__(self, E_input_channel, E_output_channels_1, E_output_channels_2, E_output_channels_3,
                 Pool_1, Pool_2, Pool_3, feature_map_size):
        super().__init__()
        self.encoder_net = nn.Sequential(
            nn.Conv2d(E_input_channel, E_output_channels_1, kernel_size=3, padding=1),
            FEM(in_channels=E_output_channels_1, num_reduced_channels=1,
                 feature_map_size=feature_map_size, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(Pool_1),
            nn.Conv2d(E_output_channels_1, E_output_channels_2, kernel_size=3, padding=1),
            FEM(in_channels=E_output_channels_2, num_reduced_channels=1,
                 feature_map_size=(feature_map_size[0] // Pool_1[0], feature_map_size[1] // Pool_1[1]), kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(Pool_2),

            nn.Conv2d(E_output_channels_2, E_output_channels_3, kernel_size=3, padding=1),
            FEM(in_channels=E_output_channels_3, num_reduced_channels=1,
                 feature_map_size=(feature_map_size[0] // (Pool_1[0] * Pool_2[0]),
                                   feature_map_size[1] // (Pool_1[1] * Pool_2[1])), kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(Pool_3)
        )

    def forward(self, x):
        return self.encoder_net(x)

class Decoder(nn.Module):
    def __init__(self, D_input_channel, D_output_channels_1, D_output_channels_2, D_output_channels_3,
                 Up_Pool_1, Up_Pool_2, Up_Pool_3, feature_map_size):
        super().__init__()
        self.decoder_net = nn.Sequential(
            nn.Upsample(scale_factor=Up_Pool_1),
            nn.ConvTranspose2d(D_input_channel, D_output_channels_1, kernel_size=3, padding=1),
            nn.ReLU(),
            FEM(in_channels=D_output_channels_1, num_reduced_channels=1,
                 feature_map_size=(feature_map_size[0] * Up_Pool_1[0], feature_map_size[1] * Up_Pool_1[1]), kernel_size=3),

            nn.Upsample(scale_factor=Up_Pool_2),
            nn.ConvTranspose2d(D_output_channels_1, D_output_channels_2, kernel_size=3, padding=1),
            nn.ReLU(),
            FEM(in_channels=D_output_channels_2, num_reduced_channels=1,
                 feature_map_size=(feature_map_size[0] * Up_Pool_1[0] * Up_Pool_2[0],
                                   feature_map_size[1] * Up_Pool_1[1] * Up_Pool_2[1]), kernel_size=3),

            nn.Upsample(scale_factor=Up_Pool_3),
            nn.ConvTranspose2d(D_output_channels_2, D_output_channels_3, kernel_size=3, padding=1),
            nn.Sigmoid(),
            FEM(in_channels=D_output_channels_3, num_reduced_channels=1,
                 feature_map_size=(feature_map_size[0] * Up_Pool_1[0] * Up_Pool_2[0] * Up_Pool_3[0],
                                   feature_map_size[1] * Up_Pool_1[1] * Up_Pool_2[1] * Up_Pool_3[1]), kernel_size=3),
        )

    def forward(self, x):
        return self.decoder_net(x)

class CAE(nn.Module):
    def __init__(self,
                 E_input_channel, E_output_channels_1, E_output_channels_2, E_output_channels_3,
                 D_input_channel, D_output_channels_1, D_output_channels_2, D_output_channels_3,
                 feature_map_size,
                 Pool_1, Pool_2, Pool_3,
                 Up_Pool_1, Up_Pool_2, Up_Pool_3,
                 lowdim_feature):
        super().__init__()

        # 三层Encoder
        self.encoder = Encoder(
            E_input_channel, E_output_channels_1, E_output_channels_2, E_output_channels_3,
            Pool_1, Pool_2, Pool_3,
            feature_map_size
        )

        # 三层Decoder
        self.decoder = Decoder(
            D_input_channel, D_output_channels_1, D_output_channels_2, D_output_channels_3,
            Up_Pool_1, Up_Pool_2, Up_Pool_3,
            feature_map_size=(feature_map_size[0] // (Pool_1[0] * Pool_2[0] * Pool_3[0]),
                              feature_map_size[1] // (Pool_1[1] * Pool_2[1] * Pool_3[1]))
        )

        # 自动计算flatten后的大小
        final_H = feature_map_size[0] // (Pool_1[0] * Pool_2[0] * Pool_3[0])
        final_W = feature_map_size[1] // (Pool_1[1] * Pool_2[1] * Pool_3[1])
        self.flatten_dim = E_output_channels_3 * final_H * final_W

        self.fc1 = nn.Linear(self.flatten_dim, lowdim_feature)
        self.fc2 = nn.Linear(lowdim_feature, self.flatten_dim)

        self.reshape_C = E_output_channels_3
        self.reshape_H = final_H
        self.reshape_W = final_W

    def forward(self, x):
        In = x.float()
        Encoder_out = self.encoder(In)
        #print(Encoder_out.shape)
        Flatten = torch.flatten(Encoder_out, start_dim=1)
        #print(Flatten.shape)
        encoder_out = F.relu(self.fc1(Flatten))
        #print(encoder_out.shape)
        Decoder_in = F.relu(self.fc2(encoder_out))
        #print(Decoder_in.shape)
        Decoder_in = Decoder_in.view(-1, self.reshape_C, self.reshape_H, self.reshape_W)
        decoder_out = self.decoder(Decoder_in)
        return encoder_out, decoder_out
    
device = torch.device("cuda")
from torchsummary import summary
model = cae = CAE(
    E_input_channel=3,
    E_output_channels_1=3,
    E_output_channels_2=3,
    E_output_channels_3=3,
    D_input_channel=3,
    D_output_channels_1=3,
    D_output_channels_2=3,
    D_output_channels_3=3,
    feature_map_size=(100, 120),
    Pool_1=(2,2), Pool_2=(2,2), Pool_3=(1,1),
    Up_Pool_1=(1,1), Up_Pool_2=(2,2), Up_Pool_3=(2,2),
    lowdim_feature=16
).to(device)
summary(model, input_size=(3, 100, 120))
from torch.utils.data import Dataset,DataLoader
class MyDataset(Dataset):
    def __init__(self, input,tgt):
        self.input = input
        self.tgt = tgt


    def __getitem__(self, index):

        return self.input[index], self.tgt[index]

    def __len__(self):
        return self.input.shape[0]
batch_size = 16
train_dataset = MyDataset(x_train,x_train)
test_dataset = MyDataset(x_train,x_train)
val_dataset = MyDataset(x_val,x_val)
train_loader = DataLoader(train_dataset,batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)