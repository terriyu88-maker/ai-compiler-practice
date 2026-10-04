


def forward(self, x):
    conv1 = self.conv1(x);  x = None
    bn1 = self.bn1(conv1);  conv1 = None
    relu = torch.nn.functional.relu(bn1, inplace = False);  bn1 = None
    max_pool2d = torch.nn.functional.max_pool2d(relu, 2, stride = None, padding = 0, dilation = 1, ceil_mode = False, return_indices = False);  relu = None
    conv2 = self.conv2(max_pool2d);  max_pool2d = None
    bn2 = self.bn2(conv2);  conv2 = None
    relu_1 = torch.nn.functional.relu(bn2, inplace = False);  bn2 = None
    max_pool2d_1 = torch.nn.functional.max_pool2d(relu_1, 2, stride = None, padding = 0, dilation = 1, ceil_mode = False, return_indices = False);  relu_1 = None
    flatten = max_pool2d_1.flatten(1);  max_pool2d_1 = None
    fc1 = self.fc1(flatten);  flatten = None
    relu_2 = torch.nn.functional.relu(fc1, inplace = False);  fc1 = None
    fc2 = self.fc2(relu_2);  relu_2 = None
    relu_3 = torch.nn.functional.relu(fc2, inplace = False);  fc2 = None
    fc3 = self.fc3(relu_3);  relu_3 = None
    return fc3
    