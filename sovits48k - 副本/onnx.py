import torch.nn as nn
import torch.nn.functional as F
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LeNet(nn.Module):
    def __init__(self, class_num=10, input_shape=(1, 32, 32)):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(  # input_size=(1*28*28)
            nn.Conv2d(1, 6, 5, 1, 2),  # padding=2保证输入输出尺寸相同
            nn.ReLU(),  # input_size=(6*28*28)
            nn.MaxPool2d(kernel_size=2, stride=2),  # output_size=(6*14*14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),  # padding=1输出尺寸变化
            nn.ReLU(),  # input_size=(16*10*10)
            nn.MaxPool2d(2, 2)  # output_size=(16*5*5)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(16 * ((input_shape[1] // 2 - 4) // 2) * ((input_shape[2] // 2 - 4) // 2), 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, class_num)

        # 定义前向传播过程，输入为x

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)


input_shape = (1,100,100)   #输入数据
model = LeNet(input_shape=input_shape)

torch.save(model, './pth/meaqua.pth')
torch_model = torch.load("./pth/meaqua.pth") # pytorch模型加载
batch_size = 1  #批处理大小


# set the model to inference mode
torch_model.eval（)

x = torch.randn(batch_size,*input_shape)        # 生成张量
print (x.shape)
export_onnx_file = "meaqua.onnx"              # 目的ONNX文件名
torch.onnx.export(torch_model,
                   x,
                   export_onnx_file,
                   opset_version=10,
                   do_constant_folding=True,  # 是否执行常量折叠优化
                   input_names=["input"],   # 输入名
                   output_names=["output"], # 输出名
                   dynamic_axes={"input":{0:"batch_size"},    # 批处理变量
                                   "output":{0:"batch_size"}})
