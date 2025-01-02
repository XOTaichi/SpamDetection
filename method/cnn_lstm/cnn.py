import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, embedding_dim, vocab_size, num_filters, kernel_size, hidden_dim, dropout_keep_prob, num_classes):
        super(CNN, self).__init__()
        self.embeds = nn.Embedding(vocab_size, embedding_dim)
        self.conv = nn.Conv1d(
            in_channels = embedding_dim,
            out_channels=num_filters,
            kernel_size=kernel_size
        )
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.dense = nn.Linear(num_filters, hidden_dim)
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(hidden_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.embeds(x)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.maxpool(x)

        x = x.view(x.size(0), -1)

        x = self.dense(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.softmax(x)
        return x