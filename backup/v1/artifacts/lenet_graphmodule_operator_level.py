


def forward(self, x):
    conv1_weight = self.conv1.weight
    conv1_bias = self.conv1.bias
    conv2d = torch.conv2d(x, conv1_weight, conv1_bias, (1, 1), (2, 2), (1, 1), 1);  x = conv1_weight = conv1_bias = None
    relu = torch.nn.functional.relu(conv2d, inplace = False);  conv2d = None
    max_pool2d = torch.nn.functional.max_pool2d(relu, 2, stride = None, padding = 0, dilation = 1, ceil_mode = False, return_indices = False);  relu = None
    conv2_weight = self.conv2.weight
    conv2_bias = self.conv2.bias
    conv2d_1 = torch.conv2d(max_pool2d, conv2_weight, conv2_bias, (1, 1), (0, 0), (1, 1), 1);  max_pool2d = conv2_weight = conv2_bias = None
    relu_1 = torch.nn.functional.relu(conv2d_1, inplace = False);  conv2d_1 = None
    max_pool2d_1 = torch.nn.functional.max_pool2d(relu_1, 2, stride = None, padding = 0, dilation = 1, ceil_mode = False, return_indices = False);  relu_1 = None
    flatten = max_pool2d_1.flatten(1);  max_pool2d_1 = None
    fc1_weight = self.fc1.weight
    fc1_bias = self.fc1.bias
    linear = torch._C._nn.linear(flatten, fc1_weight, fc1_bias);  flatten = fc1_weight = fc1_bias = None
    relu_2 = torch.nn.functional.relu(linear, inplace = False);  linear = None
    fc2_weight = self.fc2.weight
    fc2_bias = self.fc2.bias
    linear_1 = torch._C._nn.linear(relu_2, fc2_weight, fc2_bias);  relu_2 = fc2_weight = fc2_bias = None
    relu_3 = torch.nn.functional.relu(linear_1, inplace = False);  linear_1 = None
    fc3_weight = self.fc3.weight
    fc3_bias = self.fc3.bias
    linear_2 = torch._C._nn.linear(relu_3, fc3_weight, fc3_bias);  relu_3 = fc3_weight = fc3_bias = None
    return linear_2
    