


def forward(self, x):
    block1_conv = self.block1.conv(x);  x = None
    block1_bn = self.block1.bn(block1_conv);  block1_conv = None
    relu = torch.nn.functional.relu(block1_bn, inplace = False);  block1_bn = None
    block2_conv = self.block2.conv(relu);  relu = None
    block2_bn = self.block2.bn(block2_conv);  block2_conv = None
    relu_1 = torch.nn.functional.relu(block2_bn, inplace = False);  block2_bn = None
    max_pool2d = torch.nn.functional.max_pool2d(relu_1, 2, stride = None, padding = 0, dilation = 1, ceil_mode = False, return_indices = False);  relu_1 = None
    flatten = max_pool2d.flatten(1);  max_pool2d = None
    fc = self.fc(flatten);  flatten = None
    return fc
    