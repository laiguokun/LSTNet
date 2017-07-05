import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        self.use_cuda = args.cuda
        self.P = args.window;
        self.m = data.m
        self.hidR = args.hidRNN;
        self.hidC = args.hidCNN;
        self.hidS = args.hidSkip;
        self.Ck = args.CNN_kernel;
        self.skip = args.skip;
        self.pt = (self.P - self.Ck)/self.skip
        self.hw = args.highway_window
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size = (self.Ck, self.m));
        self.GRU1 = nn.GRU(self.hidC, self.hidR);
        self.GRUskip = nn.GRU(self.hidC, self.hidS);
        self.dropout = nn.Dropout(p = args.dropout);
        self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.m);
        self.highway = nn.Linear(self.hw, 1);
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid;
        else:
            self.output = None;
 
    def forward(self, x):
        batch_size = x.size(0);
        
        #CNN
        c = x.view(-1, 1, self.P, self.m);
        c = F.relu(self.conv1(c));
        c = self.dropout(c);
        c = torch.squeeze(c, 3);
        
        # RNN 
        r = c.permute(2, 0, 1).contiguous();
        _, r = self.GRU1(r);
        r = self.dropout(torch.squeeze(r,0));

        
        #skip-rnn
        s = c[:,:, -self.pt * self.skip:].contiguous();
        s = s.view(batch_size, self.hidC, self.pt, self.skip);
        s = s.permute(2,0,3,1).contiguous();
        s = s.view(self.pt, batch_size * self.skip, self.hidC);
        _, s = self.GRUskip(s);
        s = s.view(batch_size, self.skip * self.hidS);
        s = self.dropout(s);
        r = torch.cat((r,s),1);
        
        r = self.linear1(r);
        
        #highway
        z = x[:, -self.hw:, :];
        z = z.permute(0,2,1).contiguous().view(-1, self.hw);
        z = self.highway(z);
        z = z.view(-1,self.m);
        
        res = r + z;
        if (self.output):
            res = self.output(r+z);
        return res;
    
        
        
        