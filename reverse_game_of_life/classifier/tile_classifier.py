import numpy as np

from .classifier import Classifier
from ..conway_board import ALIVE,DEAD
from ..tools import board_to_int,int_to_board

from collections import Counter


class TileClassifier(Classifier):
    ''' Use info about tiles/subboards to do classification. '''

    def __init__(self,tile_size):
        self.tile_size = tile_size
        self.tiles = dict()
        self.counts = dict()

    
    def clear_tiles(self):
        ''' Removes all stored tiles. '''
        self.tiles = dict()
        self.counts = dict()
        
        
    def store_tiles(self, examples):
        ''' Add tiles from these examples. Note, occurence counts are used so recommended to not store the same examples multiple times without calling clear_tiles() in between. '''

        step = 1
        for example in examples:
            num_rows,num_cols = example.start_board.board.shape
            if example.delta not in self.tiles:
                self.tiles[example.delta] = dict()
                self.counts[example.delta] = dict()
                
            for i in range(num_rows-self.tile_size+1):
                for j in range(num_cols-self.tile_size+1):
                    start_int = board_to_int(example.start_board.board[i:(i+self.tile_size),j:(j+self.tile_size)])
                    end_int = board_to_int(example.end_board.board[i:(i+self.tile_size),j:(j+self.tile_size)])
                    if end_int not in self.tiles[example.delta]:
                        self.tiles[example.delta][end_int] = Counter()
                        self.counts[example.delta][end_int] = 0

                    self.tiles[example.delta][end_int].update([start_int])
                    self.counts[example.delta][end_int] += 1
            
    

    def predict(self,end_board,delta):
        if delta not in self.tiles:
            raise ValueError('No tiles stored for delta='+str(delta))
        
        
        num_rows,num_cols = end_board.board.shape
        
        weight_alive = np.zeros([num_rows,num_cols])
        weight_dead = np.zeros([num_rows,num_cols])
        
        for i in range(num_rows-self.tile_size+1):
            for j in range(num_cols-self.tile_size+1):
                end_int = board_to_int(end_board.board[i:(i+self.tile_size),j:(j+self.tile_size)])
                if end_int in self.tiles[delta]:
                    weight = 1./self.counts[delta][end_int]
                    for start_int in self.tiles[delta][end_int]:
                        subboard = int_to_board(start_int,self.tile_size,self.tile_size)
                        
                        weight_alive[i:(i+self.tile_size),j:(j+self.tile_size)] += subboard * weight
                        weight_dead[i:(i+self.tile_size),j:(j+self.tile_size)] += (1-subboard) * weight
                        
                        
        prediction = np.empty([num_rows,num_cols])
        prediction[...] = DEAD
        prediction[weight_alive>weight_dead] = ALIVE
        return prediction
