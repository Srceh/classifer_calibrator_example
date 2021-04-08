import warnings
from test_case import draw_test_case
from test_case import draw_test_case3

import os
if not os.path.exists('figures'):
    os.makedirs('figures')

cal_ref_list = ['EMP10', 'EMP32', 'ISO_L', 'ISO_S', 'PLATT', 'BETA']

for cal_ref in cal_ref_list:
    print(cal_ref)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(1, 4):
            print('========================================')
            print('drawing: ' + str(i))
            draw_test_case(i, cal_ref)


cal_ref_list = ['TS', 'VS', 'MS', 'DIR']

for cal_ref in cal_ref_list:
    print(cal_ref)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        draw_test_case3(cal_ref)
