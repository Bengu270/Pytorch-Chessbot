# Pytorch-Chessbot
A PyTorch CNN that predicts the next chess move from a board position, trained on about 3.9 million positions from 2000+ Elo Lichess games.
--- Currently in prototype stage ---

# How it works

Input: a 17x8x8 tensor made of 12 piece planes, 1 side-to-move plane and 4 castling-rights planes.

Output: 4032 classes, one for every pair of different squares (64 x 63).

Playing: illegal moves are masked out with python-chess, and the highest-scoring legal move is played. Pawn promotions default to queen.

# Training

Loss: CrossEntropyLoss
Optimizer: Adam
Batch size: 512

# Requirements

torch, numpy, python-chess
