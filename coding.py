''' 
在一个二维数组中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。[[1,2,8,9],[2,4,9,12],[4,7,10,13],[6,8,11,15]]给定 target = 7，返回 true。给定 target = 3，返回 false。
'''
def find_num(lists, num):
    height = len(lists)
    width = len(lists[0])
    if num < lists[0][0] or num > lists[height-1][width-1]:
        return False

    cur_j = 0
    for i in range(height):
        if num < lists[i][width-1]:
            while lists[i][cur_j] < num:
                cur_j += 1
            if lists[i][cur_j] == num:
                return True
            else:
                cur_j -= 1
        if i == height -1:
            return False
        
if __name__ == "__main__":
    lists = [[1,2,8,9],[2,4,9,12],[4,7,10,13],[6,8,11,15]]
    num = 3
    print(find_num(lists, num))
        
            