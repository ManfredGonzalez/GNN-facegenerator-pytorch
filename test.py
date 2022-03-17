import sys
sys.path.append("differential_privacy/")
from differential_privacy.laplace import LaplaceTruncated

laplace = LaplaceTruncated(epsilon=3.14,sensitivity=125,lower=-0.057,upper=0.27)
print(laplace.randomise(-0.08))