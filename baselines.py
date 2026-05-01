import math
import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

import numpy as np


STOP_WORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "he", "him", "his", "she", "her", "hers", "it", "its", "they",
    "them", "their", "what", "which", "who", "whom", "this", "that", "these",
    "those", "am", "is", "are", "was", "were", "be", "been", "being", "have",
    "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "must", "shall", "can", "a", "an", "the", "and", "but",
    "or", "nor", "for", "so", "yet", "at", "by", "in", "of", "on", "to",
    "up", "as", "into", "with", "about", "between", "through",
}


