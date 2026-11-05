import numpy as np
import torch
import matplotlib as plt
import chess


# ----- 1. Reading the data -----

def RemoveLowELO(filePath: str): # Removes all games with elo below 2000
    with open('FilteredGames.txt', 'a') as f:
        with open(filePath, 'r') as n:
            count = -1
            add = False
            for line in n:
                count +=1
                if line.startswith('[WhiteElo'):
                    if line[11] == '2':
                        add = True
                elif line.startswith('1'):
                    if add:
                        add = False
                        f.write(line)
                print(count)

# RemoveLowELO('data/games.pgn')

def CreateMovesList(filePath: str): # Removes uneccessary text to leave a list of moves per game
    with open('ListedGames.txt', 'a') as f:
        lineNum=0
        with open(filePath, 'r') as n:
            for line in n:
                line = line.replace("?", "")
                line = line.replace("!", "")
                list = line.split(' ')
                toRemove = []
                for i in range(len(list)):
                    if list[i].startswith(('{', '[', '}')):
                        toRemove.append(i)
                    if list[i].endswith((']', '.')):
                        toRemove.append(i)

                counter=0
                for item in toRemove:
                    list.pop(item - counter)
                    counter +=1

                list.pop()
                f.write(' '.join(list) + '\n')

                lineNum +=1
                print(lineNum)

# CreateMovesList('FilteredGames.txt')

def moveToIndex(fromSq: int, toSq: int):
    # Maps (from, to) to 0..4031, skipping the impossible from == to case
    return fromSq * 63 + (toSq - 1 if toSq > fromSq else toSq)

def MoveStringToMoveIndexList(moves: str): # Turns an input string of moves into a list of indexes unique for each move
    moves = moves.split()
    board = chess.Board()
    output = []
    for move in moves:
        x = board.parse_san(move)
        output.append(moveToIndex(x.from_square, x.to_square))
        board.push(x)
    return output

def ListToNumpyArrayFile(list):
    numpyArray = np.array(list)
    np.save('Y.npy', numpyArray)

def MoveStringFileToMoveIndexListFile(filePath: str): # Applies moveStringToMoveIndexList method to a file
    with open ('IndexedMovesFile.txt', 'a') as f:
        with open (filePath, 'r') as n:
            for line in n:
                list = MoveStringToMoveIndexList(line)
                for item in list:
                    f.write(str(item) + '\n')
     
# MoveStringFileToMoveIndexListFile('ListedGames.txt')

def MoveIndexListFileToNumpyArrayFile(filePath: str): # Turns a list of numbers into a numpy array file
    list = []
    count = 0
    with open (filePath, 'r') as f:
        for line in f:
            line = line.replace('\n', '')
            list.append(int(line))
            count+=1
            print(count)
    ListToNumpyArrayFile(list)

# MoveIndexListFileToNumpyArrayFile('IndexedMovesFile.txt')

def MovesStringToFenNotation(moves: str): # Turns a list of moves into a list of fen notation
    moves = moves.split()
    board = chess.Board()
    output = []
    for move in moves:
        x = board.parse_san(move)
        fen = board.fen()
        fen = fen[:len(fen)-6]
        output.append(fen)
        board.push(x)
    
    return output

def FenToNumpyArray(fen: str): # Turns an input fen string into a numpy array
    idx = fen.find(' ')
    boardState = fen[:idx]
    playerToMove = fen[idx+1]
    castleRights = fen[idx+3:]
    numpyArray = np.zeros((17,8,8), dtype=np.uint8)
    row = 0
    col = 0
    for letter in boardState:
        match letter:
            case 'p':
                numpyArray[0, row, col] = 1
                col +=1
            case 'r':
                numpyArray[1, row, col] = 1
                col +=1
            case 'n':
                numpyArray[2, row, col] = 1
                col +=1
            case 'b':
                numpyArray[3, row, col] = 1
                col +=1
            case 'q':
                numpyArray[4, row, col] = 1
                col +=1
            case 'k':
                numpyArray[5, row, col] = 1
                col +=1
            case 'P':
                numpyArray[6, row, col] = 1
                col +=1
            case 'R':
                numpyArray[7, row, col] = 1
                col +=1
            case 'N':
                numpyArray[8, row, col] = 1
                col +=1
            case 'B':
                numpyArray[9, row, col] = 1
                col +=1
            case 'Q':
                numpyArray[10, row, col] = 1
                col +=1
            case 'K':
                numpyArray[11, row, col] = 1
                col +=1
            case '/':
                row +=1
                col = 0
            case _:
                col += int(letter)

    if playerToMove == 'w':
        numpyArray[12, :, :] = 1
    else:
        numpyArray[12, :, :] = 0

    if 'k' in castleRights:
        numpyArray[13, :, :] = 1
    if 'q' in castleRights:
        numpyArray[14, :, :] = 1
    if 'K' in castleRights:
        numpyArray[15, :, :] = 1
    if 'Q' in castleRights:
        numpyArray[16, :, :] = 1

    return numpyArray

def MovesFileToFenNumpyArrayFile(filePath: str): # Apples the fen to tensor methods on a file of moves
    with open(filePath, 'r') as f:
        list = []
        count = 0
        for line in f:
            moves = MovesStringToFenNotation(line)
            for fen in moves:
                list.append(FenToNumpyArray(fen))
                count += 1
                print(count)
        numpyArray = np.stack(list, axis=0)
        np.save('X.npy', numpyArray)
# Returns an array of size (n, 17, 8, 8), where n is the number of fen notation moves

# MovesFileToFenNumpyArrayFile('ListedGames.txt')

# print(np.load('Y.npy')) # 3888192 indexed moves
# print(np.load('X.npy').shape) # numpy array of dimension [3888192, 17, 8, 8]


# # ----- 2. Creating a dataset -----
# from torch.utils.data import Dataset, DataLoader

# class MoveDataset(Dataset): 
#     def __init__ (self, xPath='X.npy', yPath='Y.npy'):
#         self.X = np.load(xPath, mmap_mode='r')
#         self.Y = np.load(yPath, mmap_mode='r')
#         assert len(self.X) == len(self.Y) # Checks the number of features and labels are the same

#     def __len__(self):
#         return len(self.X)

#     def __getitem__(self, i):
#         x = torch.from_numpy(np.array(self.X[i], dtype=np.float32)) # Features of shape [17, 8, 8]
#         y = torch.tensor(int(self.Y[i]), dtype=torch.long) # Labels (integer 64)
#         return x, y


# dataset = MoveDataset()

# #  ----- 3. Splitting data into training and testing -----
# from torch.utils.data import random_split

# trainSize = 40000
# testSize = 10000
# discardSize = len(dataset)-testSize-trainSize
# trainDS, testDS, discardDS = random_split(dataset, [trainSize, testSize, discardSize]) # Splits the dataset randomly into train and test

# trainLoader = DataLoader(
#     trainDS,
#     batch_size=512,
#     shuffle=True,
#     num_workers=0
# )

# testLoader = DataLoader(
#     testDS,
#     batch_size=512,
#     shuffle=True,
#     num_workers=0
# )

# ----- 2b. Manual batching into RAM -----

X = torch.from_numpy(np.load('X.npy'))
Y = torch.from_numpy(np.load('Y.npy')).long()

perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(0))
testIdx, valIdx, trainIdx = perm[:50_000], perm[50_000:100_000], perm[100_000:]

def batches(idx, batchSize, shuffle):
    if shuffle:
        idx = idx[torch.randperm(len(idx))] # Shuffles the indexes
    for i in range(0, len(idx), batchSize):
        batch = idx[i:i+batchSize]
        yield X[batch].to(device).float(), Y[batch].to(device)

#  ----- 4. Creating the model -----
import torch.nn as nn


class ChessBot(nn.Module):
    def __init__(self, inputShape: int, hiddenUnits: int, outputShape:int):
        super().__init__()
        self.convBlock1 = nn.Sequential(
            nn.Conv2d(inputShape, hiddenUnits, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hiddenUnits),
            nn.ReLU(),

            nn.Conv2d(hiddenUnits, hiddenUnits, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hiddenUnits),
            nn.ReLU(),

            nn.Conv2d(hiddenUnits, hiddenUnits, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hiddenUnits),
            nn.ReLU(),

            nn.Conv2d(hiddenUnits, hiddenUnits, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hiddenUnits),
            nn.ReLU()
        )
                        
        self.classifier = nn.Sequential(
            nn.Conv2d(hiddenUnits, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32*8*8, outputShape)
        )

    def forward(self, x):
        return self.classifier(self.convBlock1(x))


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model0 = ChessBot(17, 128, 4032).to(device)

# ----- 5. Optimiser and loss function

lossFn = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(params = model0.parameters(), lr = 0.001)

# ----- 6. Training and testing loops -----

def AccuracyFn(trueVal, predVal):
    correct = torch.eq(trueVal, predVal).sum().item()
    acc = (correct / len(predVal)) * 100
    return acc

def TrainStep(model: nn.Module, lossFunction: nn.Module, optimiser: torch.optim.Optimizer):
    model.train()
    trainLoss, trainAcc, numBatches = 0, 0, 0
    for x, y in batches(trainIdx, 512, shuffle=True):
        x, y = x.to(device), y.to(device)
        yLogit = model(x)
        loss = lossFunction(yLogit, y)
        trainLoss += loss
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        numBatches +=1

        yPred = yLogit.softmax(dim=1).argmax(dim=1)
        acc = AccuracyFn(y, yPred)
        trainAcc += acc

    print(f'Train loss: {trainLoss/numBatches}, Train accuracy: {trainAcc/numBatches}')
        
def TestStep(model: nn.Module, lossFunction: nn.Module):
    model.eval()
    with torch.inference_mode():
        testLoss, testAcc, numBatches = 0, 0, 0

        for x, y in batches(testIdx, 512, shuffle=True):
            x, y = x.to(device), y.to(device)
            yLogit = model(x)
            loss = lossFunction(yLogit, y)
            testLoss += loss
            yPred = yLogit.softmax(dim=1).argmax(dim=1)
            acc = AccuracyFn(y, yPred)
            testAcc += acc
            numBatches +=1
        print(f'Test loss: {testLoss/numBatches}, Test accuracy: {testAcc/numBatches}')
    return testAcc

def TrainModel(epochs):
    best = 0
    for epoch in range(epochs):
        
        print(f'epoch: {epoch}')
        TrainStep(model0, lossFn, optimiser)
        epochAcc = TestStep(model0, lossFn)
        if epochAcc > best:
            best = epochAcc
            torch.save(model0.state_dict(), 'models/chessbot.pt')

# TrainModel(10)

def LoadModel(path):
    model = ChessBot(17, 128, 4032).to(device)
    model.load_state_dict(torch.load(path, weights_only=True, map_location=device))
    return model
