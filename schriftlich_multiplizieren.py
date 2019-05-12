#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 11:43:32 2019

@author: detlef
"""
from random import randint
import numpy as np

maxlen1 = 7
maxlen2 = 5

#setup vocabulary
vocab = {}
vocab_rev = {}
count_chars = 0
def add_translate(cc):
    #pylint: disable=W0603
    global count_chars
    vocab[cc] = count_chars
    vocab_rev[count_chars] = cc
    count_chars += 1

for c in range(ord('0'), ord('9')+1):
    add_translate(chr(c))

add_translate('+')
add_translate('*')
add_translate('=')
#add_translate('.')
add_translate(' ')

print("num of different chars", len(vocab))



def one_data(maxlen1, maxlen2, debug = False):
    sequence_out = []
    sequence_in = []
    result = []
    
    xx1 = [randint(0,9) for _ in range(maxlen1)]
    x1 = 0
    for x in xx1:
       x1*=10
       x1+=x
    xx2 = [randint(0,9) for _ in range(maxlen2)]
    x2 = 0
    for x in xx2:
       x2*=10
       x2+=x
    if debug:
        print(x1*x2)
    
    len1 = len(str(x1))
    len2 = len(str(x2))
    
    if debug:
        print(len1,len2)
    
    result.append(str(x1)+"*"+str(x2)+"=")
    
    pos1 = [0,0]
    pos2 = [0,0]
    pos3 = [0,0]
    
    def movepos(p,d):
        if d == 'r':
            p[0]+=1
        elif d == 'l':
            p[0]-=1
        elif d == 'u':
            p[1]-=1
        elif d == 'd':
            p[1]+=1
#        if p[0]<0:
#            p[0]=0
#        if p[1]<0:
#            p[1]=0
    
    def get_int1():
        if debug:
            print("pos1",pos1,get_vector_pos(pos1),get_vector_pos(pos2))
        if pos1[1]<0:
            return 0
        r = result[pos1[1]][pos1[0]]
        if r == ' ':
            return 0
        return int(r)
    def get_int2():
        if debug:
            print("pos2",pos2,get_vector_pos(pos1),get_vector_pos(pos2))
        if len(result[pos2[1]]) <= pos2[0]:
            return 0
        r = result[pos2[1]][pos2[0]]
        if r == ' ':
            return 0
        return int(r)
    def get_int3():
        if debug:
            print("pos3",pos3)
        r = result[pos3[1]][pos3[0]]
        if r == ' ':
            return 0
        return int(r)
    
    def get_vector_pos(pos):
        vec = []
        if pos[1]>=0 and pos[1]<len(result)-1 and pos[0]<len(result[pos[1]]):
            vec.append(result[pos[1]][pos[0]])
        else:
            vec.append(' ')
#        if pos[1]>0 and pos[0]<len(result[pos[1]-1]):
#            vec.append(result[pos[1]-1][pos[0]])
#        else:
#            vec.append(' ')
#        if pos[1]>=0 and pos[1]<len(result)-1 and pos[0]<len(result[pos[1]+1]):
#            vec.append(result[pos[1]+1][pos[0]])
#        else:
#            vec.append(' ')
#        if pos[0]>0 and pos[0] <len(result[pos[1]]):
#            vec.append(result[pos[1]][pos[0]])
#        else:
#            vec.append(' ')
        if pos[1]>=0 and pos[0] + 1<len(result[pos[1]]):
            vec.append(result[pos[1]][pos[0]+1])
        else:
            vec.append(' ')
        return vec
        
           
    def move_or_set(direction):
        #global sequence_out
        #global seqence_in
        
        sequence_out.append(direction)
        sequence_in.append(get_vector_pos(pos1)+get_vector_pos(pos2))
        if direction == 'a':
            movepos(pos1,'r')
        elif direction == 'b':
            movepos(pos1,'l')
        elif direction == 'c':
            movepos(pos1,'d')
        elif direction == 'd':
            movepos(pos1,'u')
    
        elif direction == 'e':
            movepos(pos2,'r')
        elif direction == 'f':
            movepos(pos2,'l')
        elif direction == 'g':
            movepos(pos2,'d')
        elif direction == 'h':
            movepos(pos2,'u')
    
        elif direction == 'i':
            movepos(pos3,'r')
        elif direction == 'j':
            movepos(pos3,'l')
        elif direction == 'k':
            movepos(pos3,'d')
        elif direction == 'l':
            movepos(pos3,'u')
        else:
            while pos3[1] >= len(result):
                result.append("")
            while pos3[0] >= len(result[pos3[1]]):
                result[pos3[1]] += ' '
            result[pos3[1]] = result[pos3[1]][:pos3[0]]+direction+result[pos3[1]][pos3[0]+1:]
        #double in case of set but not move
        while pos3[1] >= len(result):
            result.append("")
        while pos3[0] >= len(result[pos3[1]]):
            result[pos3[1]] += ' '
    
    #move_or_set('a')
    #move_or_set('e')
    #move_or_set('i')
    
    for _ in range(len1-1):
        move_or_set('a')
    for _ in range(len1+2-1):
        move_or_set('e')
        move_or_set('i')
    move_or_set('k')
    
    for pp in range(len2):
        p = get_int1()*get_int2()
        c = int(p / 10)
        move_or_set(str(p)[-1:])
        if debug:
                for r in result:
                    print(r)
        for _ in range(len1-1):
            move_or_set('b')
            move_or_set('j')
            p = get_int1()*get_int2()+c
            c = int(p / 10)
            move_or_set(str(p)[-1:])
            if debug:
                for r in result:
                    print(r)
        move_or_set('j')
        move_or_set(str(c))
        move_or_set('i')
        for _ in range(len1-1):
            move_or_set('a')
            move_or_set('i')
        if pp < len2-1:
            move_or_set('i')
            move_or_set('e')
        move_or_set('k')
    move_or_set('g')
    move_or_set('d') # to be different for sum
    for pp in range(len1+len2):
        move_or_set('k')
        if debug:
                print(pos1,pos2,pos3)
        p=0
        for _ in range(len2+1):
            p += get_int2()
            move_or_set('g')
        move_or_set(str(p)[-1:])
        c = int(p / 10)
        move_or_set('j')
        if pp < len1+len2-1:
            move_or_set('l')
            move_or_set(str(c))
            if debug:
                for r in result:
                    print(r)

            for _ in range(len2+1):
                move_or_set('h')    
            move_or_set('f')
    move_or_set(' ') #end marker, spaces should never be written in other situations
    return sequence_in, sequence_out, result

sequence_in, sequence_out, result = one_data(maxlen1, maxlen2, True)

print()    

#print(sequence_out, sequence_in)

np_in = np.array(list(map((lambda x: list(map((lambda y: vocab[y]), x))), sequence_in)))
np_swap = np.swapaxes(np_in,0,1)

#print(np_in.shape, np_swap.shape)
#print(np_in)
#print(np_swap)

for r in result:
    print(r)

