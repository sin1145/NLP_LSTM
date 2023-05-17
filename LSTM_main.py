import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import argparse
from utils import Corpus,LSTMmodel

# 超参数
parser = argparse.ArgumentParser()
parser.add_argument('--embed_size', type=int, default=256,
                    help='initial value of batch size, default 256')
parser.add_argument('--hidden_size', type=int, default=1024,
                    help='initial value of epoch number, default 1024')
parser.add_argument('--num_layers', type=int, default=3,
                    help='initial value of epoch number, default 3')
parser.add_argument('--num_epochs', type=int, default=10,
                    help='initial value of time step, default 10')
parser.add_argument('--batch_size', type=int, default=50,
                    help='initial value of offset, default 50')
parser.add_argument('--seq_length', type=int, default=30,
                    help='initial value of hidden size, default 30')
parser.add_argument('--learning_rate', type=float, default=1e-3,
                    help='initial value of learning rate, default 0.001')
parser.add_argument('--num_samples', type=int, default=500,
                    help='initial value of sample number, default 500')
parser.add_argument('--whether_train', type=int, default=0,
                    help='whether to train or not, default not')

args = parser.parse_args()



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
corpus = Corpus()  
corpus.get_file('./jyxstxtqj_downcc.com')
ids = corpus.get_data(args.batch_size)  # 词id列表
vocab_size = len(corpus.dictionary)  # 词id总长


if args.whether_train:
    model = LSTMmodel(args.vocab_size, args.embed_size, args.hidden_size, args.num_layers).to(device)
    cost = nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    for epoch in range(args.num_epochs):
        states = (torch.zeros(args.num_layers, args.batch_size, args.hidden_size).to(device),
                  torch.zeros(args.num_layers, args.batch_size, args.hidden_size).to(device))  

        for i in tqdm(range(0, ids.size(1) - args.seq_length, args.seq_length)):  # 进度条
            inputs = ids[:, i:i + args.seq_length].to(device)  
            targets = ids[:, (i + 1):(i + 1) + args.seq_length].to(device) 

            states = [state.detach() for state in states]
            outputs, states = model(inputs, states)
            loss = cost(outputs, targets.reshape(-1))

            model.zero_grad()  
            loss.backward()  
            clip_grad_norm_(model.parameters(), 0.5)  
            optimizer.step()


# 保存模型与读取
    save_path = './model_path/model.pt'
    torch.save(model, save_path)
else:
    model = torch.load('./model_path/multi_model.pt')


 
# 文本生成：随机选择词作为生成段落的提示语
article = str() 
state = (torch.zeros(args.num_layers, 1, args.hidden_size).to(device),
         torch.zeros(args.num_layers, 1, args.hidden_size).to(device))  
prob = torch.ones(vocab_size) 
_input = torch.multinomial(prob, num_samples=1).unsqueeze(1).to(device)
for i in range(args.num_samples):
    output, state = model(_input, state)
    prob = output.exp()
    word_id = torch.multinomial(prob, num_samples=1).item()

    _input.fill_(word_id)
    word = corpus.dictionary.idx2word[word_id]
    word = '\n' if word == '<eos>' else word
    article += word
print(article)


# 保存
if len(corpus.file_list)>1:
    source = '_multi'
else:
    source = '_single'
txt_name = './generated/'+str(args.num_samples)+source+'.txt'
with open(txt_name, 'w', encoding="utf-8") as gen_file:
    gen_file.write(article)