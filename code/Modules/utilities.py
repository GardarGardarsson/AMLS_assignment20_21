#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 10:50:34 2020

@author: gardar
"""

# Print progress
def print_progress (iteration, total, message = '', length = 20):
    
    # Calculate percent complete
    percent = "{0:.1f}".format(iteration / total * 100)
    # Determine fill of loading bar, length is the total length
    fill = int(length * iteration / total)
    # Determine the empty space of the loading bar
    empty = ' ' * (length - fill)
    # Animate the bar with unicode character 2588, a filled block
    bar = u"\u2588" * fill + empty
    
    # Print loading bar
    print(f'\r{message} |{bar}| {percent}% ', end = '\r')
   
    # Print new line on completion
    if iteration == total: 
        print()
        

if __name__ == '__main__':
    
    import time
    
    for i in range(100):
        time.sleep(0.01)
        print_progress(i+1,100, "Progress")