import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, embedding_dim, vocab_size, hidden_dim, dropout_keep_prob, num_classes):
        super(LSTM, self).__init__()
        self.embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.droupout = nn.Dropout(dropout_keep_prob)
        self.dense = nn.Linear(hidden_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.embeds(x)
        lstm_out, (h_n, c_n) = self.lstm(x)
        x = h_n.squeeze(0)
        x = self.droupout(x)
        x = self.dense(x)
        x = self.softmax(x)
        return x