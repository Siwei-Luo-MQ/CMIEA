from openai import OpenAI
import argparse
from datetime import datetime
import os
import pandas as pd
import base64
import re



def main():
    parser = argparse.ArgumentParser(description='MM ADS Testing - Information Extraction')
    parser.add_argument('--data_path', default='E:\GitHub\CMIEA\Dataset', type=str, help='Path of MM crash dataset')
    parser.add_argument('--road_type_label', default='E:\GitHub\CMIEA\\road_type.xlsx', type=str,
                        help='Path of road type label file')
    args = parser.parse_args()

if __name__=='__main__':
    main()