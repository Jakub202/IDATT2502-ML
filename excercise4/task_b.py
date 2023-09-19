import torch
import torch.nn as nn
import numpy as np


class LongShortTermMemoryModel(nn.Module):
    def __init__(self, encoding_size, emoji_encoding_size):
        super(LongShortTermMemoryModel, self).__init__()
        self.encoding_size = encoding_size
        self.emoji_encoding_size = emoji_encoding_size
        self.lstm = nn.LSTM(self.encoding_size, 128)
        self.dense = nn.Linear(128, self.emoji_encoding_size)

    def reset(self):
        zero_state = torch.zeros(1, 1, 128)
        self.hidden_state = zero_state
        self.cell_state = zero_state

    def logits(self, x):
        out, (self.hidden_state, self.cell_state) = self.lstm(x, (self.hidden_state, self.cell_state))
        return self.dense(out.reshape(-1, 128))

    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x, y):
        logits = self.logits(x)
        last_logits = logits[-1]  # Take the last output only
        return nn.functional.cross_entropy(last_logits.unsqueeze(0), y.argmax(1))


index_to_char = [' ', 'h', 'a', 't', 'r', 'c', 'f', 'l', 'm', 'p', 's', 'o', 'n']

char_encodings = [
    [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # ' '
    [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'h'
    [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'a'
    [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 't'
    [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'r'
    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],  # 'c'
    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],  # 'f'
    [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],  # 'l'
    [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],  # 'm'
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],  # 'p'
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],  # 's'
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],  # 'o'
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]   # 'n'
]
encoding_size = len(char_encodings)

# Define your emojis and their encodings
emojis = {
    'hat': [1., 0., 0., 0., 0., 0., 0.],
    'rat': [0., 1., 0., 0., 0., 0., 0.],
    'cat': [0., 0., 1., 0., 0., 0., 0.],
    'flat': [0., 0., 0., 1., 0., 0., 0.],
    'matt': [0., 0., 0., 0., 1., 0., 0.],
    'cap': [0., 0., 0., 0., 0., 1., 0.],
    'son': [0., 0., 0., 0., 0., 0., 1.]
}

emoji_characters = {
    'hat': '🎩',
    'rat': '🐭',
    'cat': '🐱',
    'flat': '🏢',
    'matt': '👨',
    'cap': '🧢',
    'son': '👦'
}



emoji_encodings = np.eye(len(emojis))
emoji_encoding_size = len(emoji_encodings)

# Create your training data
x_train = torch.tensor([
    [[char_encodings[1]], [char_encodings[2]], [char_encodings[3]], [char_encodings[0]]],  # 'hat '
    [[char_encodings[4]], [char_encodings[2]], [char_encodings[3]], [char_encodings[0]]],  # 'rat '
    [[char_encodings[5]], [char_encodings[2]], [char_encodings[3]], [char_encodings[0]]],  # 'cat '
    [[char_encodings[6]], [char_encodings[7]], [char_encodings[2]], [char_encodings[3]]],  # 'flat'
    [[char_encodings[8]], [char_encodings[2]], [char_encodings[3]], [char_encodings[8]]],  # 'matt'
    [[char_encodings[5]], [char_encodings[2]], [char_encodings[9]], [char_encodings[0]]],  # 'cap '
    [[char_encodings[10]], [char_encodings[11]], [char_encodings[12]], [char_encodings[0]]]  # 'son '
], dtype=torch.float)

y_train = torch.tensor([
    [emojis['hat']],
    [emojis['rat']],
    [emojis['cat']],
    [emojis['flat']],
    [emojis['matt']],
    [emojis['cap']],
    [emojis['son']]
], dtype=torch.long)





# Initialize the model
model = LongShortTermMemoryModel(encoding_size, emoji_encoding_size)

# Initialize the optimizer
optimizer = torch.optim.RMSprop(model.parameters(), 0.001)


# Training loop
for epoch in range(500):
    for i in range(x_train.size()[0]):
        model.reset()
        model.zero_grad()
        loss = model.loss(x_train[i], y_train[i])
        loss.backward()
        optimizer.step()



# Test the model
def generate_emoji(string):
    index_to_emoji = {tuple(v): k for k, v in emojis.items()}

    model.reset()
    input_tensor = []
    for char in string:
        if char in index_to_char:
            char_index = index_to_char.index(char)
            input_tensor.append([char_encodings[char_index]])
    input_tensor = torch.tensor(input_tensor, dtype=torch.float)
    y_output = model.f(input_tensor)
    y_output = y_output.detach().numpy()  # Convert tensor to numpy array
    predicted_encoding = np.round(y_output[-1])  # Round to get one-hot encoding
    predicted_emoji = index_to_emoji.get(tuple(predicted_encoding), "Unknown")
    print(emoji_characters[predicted_emoji])

# Test the function
generate_emoji('rts')
generate_emoji('rats')
generate_emoji('hat')
generate_emoji('fat')


