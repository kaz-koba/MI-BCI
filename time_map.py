import sys

def mapping_time(x, task_num):
    if x == 1:
        time_map = [
            (0., 1., task_num*0),
            (1., 2., task_num*0),
            (2., 3., task_num*1),
            (3., 4., task_num*1)
        ]

    elif x == 2:
        time_map = [
            (0., 1., task_num*0),
            (1., 2., task_num*1),
            (2., 3., task_num*2),
            (3., 4., task_num*3),
            (4., 5., task_num*4)
        ]
    
    elif x == 3:
        time_map = [
            (0., 1., task_num*0),
            (0.5, 1.5, task_num*1),
            (1., 2., task_num*2),
            (1.5, 2.5, task_num*3),
            (2., 3., task_num*4),
            (2.5, 3.5, task_num*5),
            (3., 4., task_num*6),
            (3.5, 4.5, task_num*7),
            (4., 5., task_num*8)
        ]

    elif x == 4:
        time_map = [
            (0., 1., task_num*0),
            (0.25, 1.25, task_num*1),
            (0.5, 1.5, task_num*2),
            (0.75, 1.75, task_num*3),
            (1., 2., task_num*4),
            (1.25, 2.25, task_num*5),
            (1.5, 2.5, task_num*6),
            (1.75, 2.75, task_num*7),
            (2., 3., task_num*8),
            (2.25, 3.25, task_num*9),
            (2.5, 3.5, task_num*10),
            (2.75, 3.75, task_num*11),
            (3., 4., task_num*12),
            (3.25, 4.25, task_num*13),
            (3.5, 4.5, task_num*14),
            (3.75, 4.75, task_num*15),
            (4., 5., task_num*16)
        ]

    elif x== 5:
        time_map = [
            (0., 1., task_num*0),
            (0.5, 1.5, task_num*0),
            (1., 2., task_num*1),
            (1.5, 2.5, task_num*1),
            (2., 3., task_num*2),
            (2.5, 3.5, task_num*2),
            (3., 4., task_num*3),
            (3.5, 4.5, task_num*3)
        ]
    
    elif x == 6:
        time_map = [
            (0., 1., task_num*0),
            (0.25, 1.25, task_num*0),
            (0.5, 1.5, task_num*1),
            (0.75, 1.75, task_num*1),
            (1., 2., task_num*2),
            (1.25, 2.25, task_num*2),
            (1.5, 2.5, task_num*3),
            (1.75, 2.75, task_num*3),
            (2., 3., task_num*4),
            (2.25, 3.25, task_num*4),
            (2.5, 3.5, task_num*5),
            (2.75, 3.75, task_num*5),
            (3., 4., task_num*6),
            (3.25, 4.25, task_num*6),
            (3.5, 4.5, task_num*7),
            (3.75, 4.75, task_num*7)
        ]

    elif x==7:
        time_map = [
            (0.5, 1.5, task_num*0)
        ]


    
    else:
        print('Error: time is', file=sys.stderr)
        sys.exit(1)  

    return time_map
    