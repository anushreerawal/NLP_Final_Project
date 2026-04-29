import math
import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

import numpy as np


STOP_WORDS = {
    "i", "me", "myself", "we", "our", "ours", "ourselves", "you", "your","yours", "he", "him", "his"
}