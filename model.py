import torch
from torch import nn

# Won't need this after this PR is merged in:
# https://github.com/pytorch/pytorch/pull/22245
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Net(nn.Module):

    def __init__(self, args, device):
        super(Net, self).__init__()
        # Create a vanilla PyTorch neural network that will be
        # automatically monkey-patched by higher later.
        # Before higher, models could *not* be created like this
        # and the parameters needed to be manually updated and copied
        # for the updates.
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 3),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            Flatten(),
            nn.Linear(64, args.n_way)).to(device)

    def forward(self, x):
        return self.net(x)

    

class ICM(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim, fwd_hidden, device):
        """We model the icm module in that way of meta learning exploration.
        input dimension is the size of the paramters.
        hidden dimension is the size of the hidden layer, more a another projection of the state.
        output dimension is the size of the gradient.

        The forward model takes the hidden layer and predict the next state:
        - f(phi(s_t), gradient) -> phi(s_t+1)

        The inverse model takes the hidden layer and the next state and predict the gradient:
        - g(phi(s_t), phi(s_t+1)) -> gradient

        Args:
            input_dim (tensor): paramters size of the model (consider its should be flattened to be a vector)
            hidden_dim (tensor): paramter is transformed to a hidden layer.
            action_dim (tensor): the size of the gradient in the hidden space.
        """
        super(ICM, self).__init__()

        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), # [batch_size, input_dim] -> [batch_size, hidden_dim], i.e. reduce the size of the input.
            nn.ReLU()
        ).to(device)

        # keep in mind that the input_dim is the size of the state, and the output_dim is the size of one single action
        self.forward_model = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1024)
        ).to(device)

        # the output_dim is the size of the gradient
        self.inverse_model = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        ).to(device)

    def forward(self, s_t, action, s_t1):
        """Here we define the forward and inverse model of the ICM module.

        Args:
            s_t (_type_): the state at time t (here the parameters of the model)
            action (_type_): the gradient of the model, interpreted as an action
            s_t1 (_type_): the state at time t+1 (here the parameters of the model)

        Returns:
            _type_: the forward and backward error of the model
        """
        assert s_t.shape[0] == s_t1.shape[0]
        assert s_t.shape[0] == action.shape[0]

        s_t_emb = self.embedding(s_t)
        s_t1_emb = self.embedding(s_t1)
        action_emb = self.embedding(action)

        assert s_t_emb.shape[0] == s_t1_emb.shape[0]
        assert s_t_emb.shape[0] == action_emb.shape[0]

        predicted_state = self.forward_model(torch.cat([s_t_emb, action_emb], dim=-1))
        predicted_action = self.inverse_model(torch.cat([s_t_emb, s_t1_emb], dim=-1))

        # print(predicted_state.shape, predicted_action.shape)

        forward_error = torch.norm(s_t1_emb - predicted_state, dim=-1, p=2, keepdim=True)
        backward_error = torch.norm(action_emb - predicted_action, dim=-1, p=2, keepdim=True)

        # print(forward_error, backward_error)

        return forward_error, backward_error

class ICM_Helper(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super(ICM_Helper, self).__init__()

        # initialize the ICM model
        self.icm = ICM(input_dim, hidden_dim, action_dim)

        # initialize the optimizer
        self.optimizer = torch.optim.Adam(self.icm.parameters(), lr=1e-3)