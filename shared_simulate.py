# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 14:32:49 2019

@author: ddjan
"""

import os
import json
import syncthing_test

num_computers = 3

with open("/home/djanke3/Documents/computer_identity.json", 'r') as ci:
    computer_identity = json.load(ci)
    
def get_finished(filename, this_comp = computer_identity, total = 1):
    this_trials = read_trials(filename, this_comp, 1)
    this_trials = {k:v for k,v in this_trials.items() if str(v) != str(this_comp)}
    other_trials = read_trials(filename, (this_comp + 1) % num_computers, total - 1)
    finished = set(list(this_trials.keys()) + list(other_trials.keys()))
    #finished.remove("trial100_1200")
    # if len(this_trials) == 0:
        # input("Trials are empty. Continue?")
    return finished, this_trials

def read_trials(filename, comp_num, total):
    all_trials = {}
    for i in range(total):
        filename_num = "{}.".join(filename.split('.')).format((comp_num + i) % num_computers)
        try:
            with open(filename_num, 'r', encoding='utf-8-sig') as jf:
                trials = json.load(jf)
        except FileNotFoundError:
            trials = {}
        all_trials.update(trials)
    return all_trials


def write_trials(trials, filename, comp_num):
    filename = "{}.".join(filename.split('.')).format(comp_num)
    #print(filename)
    #if comp_num != 2: syncthing_test.force_sync()
    with open(filename, 'w') as jf:
        json.dump(trials, jf)