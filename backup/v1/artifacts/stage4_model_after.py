


def forward(self, x):
    fused_conv_bn_relu_0 = self.fused_conv_bn_relu_0(x);  x = None
    fused_conv_bn_relu_1 = self.fused_conv_bn_relu_1(fused_conv_bn_relu_0);  fused_conv_bn_relu_0 = None
    max_pool2d = torch.nn.functional.max_pool2d(fused_conv_bn_relu_1, 2, stride = None, padding = 0, dilation = 1, ceil_mode = False, return_indices = False);  fused_conv_bn_relu_1 = None
    flatten = max_pool2d.flatten(1);  max_pool2d = None
    fc = self.fc(flatten);  flatten = None
    return fc
    