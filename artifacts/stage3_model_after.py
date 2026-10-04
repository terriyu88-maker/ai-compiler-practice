


def forward(self, x):
    fused_conv_bn_relu_0 = self.fused_conv_bn_relu_0(x);  x = None
    max_pool2d = torch.nn.functional.max_pool2d(fused_conv_bn_relu_0, 2, stride = None, padding = 0, dilation = 1, ceil_mode = False, return_indices = False);  fused_conv_bn_relu_0 = None
    fused_conv_bn_relu_1 = self.fused_conv_bn_relu_1(max_pool2d);  max_pool2d = None
    max_pool2d_1 = torch.nn.functional.max_pool2d(fused_conv_bn_relu_1, 2, stride = None, padding = 0, dilation = 1, ceil_mode = False, return_indices = False);  fused_conv_bn_relu_1 = None
    flatten = max_pool2d_1.flatten(1);  max_pool2d_1 = None
    fc1 = self.fc1(flatten);  flatten = None
    relu_2 = torch.nn.functional.relu(fc1, inplace = False);  fc1 = None
    fc2 = self.fc2(relu_2);  relu_2 = None
    relu_3 = torch.nn.functional.relu(fc2, inplace = False);  fc2 = None
    fc3 = self.fc3(relu_3);  relu_3 = None
    return fc3
    