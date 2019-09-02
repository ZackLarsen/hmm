import viterbi_cython
import time

number = 10000

start = time.time()
viterbi_cython.test(number)
end =  time.time()

cy_time = end - start
print("Cython time = {}".format(cy_time))