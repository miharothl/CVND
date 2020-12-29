import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.embeding = nn.Embedding(num_embeddings=vocab_size,
                                     embedding_dim=embed_size)
        
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            batch_first = True,
                            dropout = 0.5,
                            num_layers = num_layers)
    
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        captions = captions[:, :-1]
        
        # embed the captions
        embeds = self.embeding(captions)
        x = torch.cat((features.unsqueeze( dim = 1), embeds), dim = 1) 
        
        x, _ = self.lstm(x)
        x = self.fc(x)
            
        return x

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "

        """ Generate captions with greedy search """
        predicted_sentence = []

        for i in range(max_len):
            # Get output and states from LSTM layer
            x, states = self.lstm(inputs, states)
            x = x.squeeze(1)

            # Get output of the linear layer
            x = self.fc(x)

            # Get the best predicted
            predicted = x.max(1)[1]

            # Append predicted item to predicted sentence
            predicted_sentence.append(predicted.item())
            # Update input for next sequence
            inputs = self.embeding(predicted).unsqueeze(1)

        return predicted_sentence
